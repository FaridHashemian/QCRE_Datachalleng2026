from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from config import (
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
    UNLABELED_DIR,
    ROI_TRAIN_DIR,
    ROI_VAL_DIR,
    ROI_TEST_DIR,
    ROI_UNLABELED_DIR,
    ROI_BOX_DEBUG_DIR,
    FALLBACK_TEMPLATE_PATH,
    ROI_SIZE,
    NUM_ROIS,
    EXPECTED_BOXES_PER_HALF,
    EXPECTED_TOTAL_BOXES,
    BRIGHT_ENHANCE_SIGMA,
    DOT_THRESHOLD,
    DOT_MIN_W,
    DOT_MAX_W,
    DOT_MIN_H,
    DOT_MAX_H,
    DOT_MIN_AREA,
    MERGE_DILATE_KERNEL,
    MERGE_CLOSE_KERNEL,
    PANEL_MIN_W,
    PANEL_MIN_H,
    PANEL_MIN_ASPECT,
    PANEL_MAX_ASPECT,
    PANEL_MIN_AREA_FRAC,
    PANEL_MAX_AREA_FRAC,
    OVERLAP_IOU_THRESHOLD,
    EXPAND_PAD_X,
    EXPAND_PAD_Y,
    ensure_project_dirs,
)


# =============================================================================
# Low-level detector
# =============================================================================

def _compute_iou(box_a, box_b) -> float:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter + 1e-6
    return inter / union


def _detect_dot_panels_in_half(
    half: np.ndarray,
    expected_n: int = EXPECTED_BOXES_PER_HALF,
):
    """
    Detect bright-dot perforation panels inside one half of the image.

    Returns:
        boxes: list[(x, y, w, h)] ordered top-to-bottom
        confidence: float
        stats_dict: dict
    """
    H, W = half.shape

    # Remove smooth background, keep local bright structures
    bg = cv2.GaussianBlur(
        half,
        (0, 0),
        sigmaX=BRIGHT_ENHANCE_SIGMA,
        sigmaY=BRIGHT_ENHANCE_SIGMA,
    )
    enh = cv2.subtract(half, bg)

    # Threshold bright dots
    thr = (enh > DOT_THRESHOLD).astype(np.uint8) * 255

    # Keep only small connected components = likely bright dots
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thr, connectivity=8)
    dot_mask = np.zeros_like(thr)

    kept_dot_count = 0
    kept_dot_area = 0

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if DOT_MIN_W <= w <= DOT_MAX_W and DOT_MIN_H <= h <= DOT_MAX_H and area >= DOT_MIN_AREA:
            dot_mask[labels == i] = 255
            kept_dot_count += 1
            kept_dot_area += int(area)

    # Merge nearby dots to form panel candidates
    merged = cv2.dilate(dot_mask, np.ones((MERGE_DILATE_KERNEL, MERGE_DILATE_KERNEL), np.uint8), iterations=1)
    merged = cv2.morphologyEx(
        merged,
        cv2.MORPH_CLOSE,
        np.ones((MERGE_CLOSE_KERNEL, MERGE_CLOSE_KERNEL), np.uint8),
    )

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(merged, connectivity=8)

    candidates = []
    half_area = H * W

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]

        if w < PANEL_MIN_W or h < PANEL_MIN_H:
            continue

        aspect = w / (h + 1e-6)
        area_frac = area / half_area

        if not (PANEL_MIN_ASPECT <= aspect <= PANEL_MAX_ASPECT):
            continue
        if not (PANEL_MIN_AREA_FRAC <= area_frac <= PANEL_MAX_AREA_FRAC):
            continue

        score = float(area)
        candidates.append((score, x, y, w, h))

    candidates.sort(reverse=True)

    chosen = []
    for score, x, y, w, h in candidates:
        ok = True
        for _, cx, cy, cw, ch in chosen:
            if _compute_iou((x, y, w, h), (cx, cy, cw, ch)) > OVERLAP_IOU_THRESHOLD:
                ok = False
                break
        if ok:
            chosen.append((score, x, y, w, h))
        if len(chosen) == expected_n:
            break

    boxes = sorted([(x, y, w, h) for _, x, y, w, h in chosen], key=lambda b: b[1])

    # Confidence summary for this half
    top_scores = [c[0] for c in chosen]
    confidence = 0.0
    if len(top_scores) > 0:
        confidence = float(sum(top_scores)) / float(half_area)

    stats_dict = {
        "kept_dot_count": kept_dot_count,
        "kept_dot_area": kept_dot_area,
        "num_candidates": len(candidates),
        "num_chosen": len(boxes),
        "half_confidence": confidence,
    }

    return boxes, confidence, stats_dict


def _expand_box(
    x: int,
    y: int,
    w: int,
    h: int,
    img_shape: tuple[int, int],
    pad_x: int = EXPAND_PAD_X,
    pad_y: int = EXPAND_PAD_Y,
) -> tuple[int, int, int, int]:
    H, W = img_shape

    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(W, x + w + pad_x)
    y1 = min(H, y + h + pad_y)

    return x0, y0, x1 - x0, y1 - y0


def detect_roi_boxes(gray: np.ndarray):
    """
    Returns:
        boxes: (4, 4) int32 array [x, y, w, h] in original image coordinates
        detect_ok: bool
        meta: dict

    Order:
        left-top, left-bottom, right-top, right-bottom
    """
    H, W = gray.shape
    mid = W // 2

    left = gray[:, :mid]
    right = gray[:, mid:]

    left_boxes, left_conf, left_stats = _detect_dot_panels_in_half(left, expected_n=EXPECTED_BOXES_PER_HALF)
    right_boxes, right_conf, right_stats = _detect_dot_panels_in_half(right, expected_n=EXPECTED_BOXES_PER_HALF)

    left_boxes = [_expand_box(x, y, w, h, left.shape) for (x, y, w, h) in left_boxes]
    right_boxes = [_expand_box(x, y, w, h, right.shape) for (x, y, w, h) in right_boxes]

    boxes = []

    for x, y, w, h in left_boxes:
        boxes.append([x, y, w, h])

    for x, y, w, h in right_boxes:
        boxes.append([x + mid, y, w, h])

    detect_ok = (len(left_boxes) == EXPECTED_BOXES_PER_HALF) and (len(right_boxes) == EXPECTED_BOXES_PER_HALF)

    while len(boxes) < EXPECTED_TOTAL_BOXES:
        boxes.append([0, 0, 0, 0])

    meta = {
        "left_confidence": float(left_conf),
        "right_confidence": float(right_conf),
        "total_confidence": float(left_conf + right_conf),
        "left_stats": left_stats,
        "right_stats": right_stats,
        "detect_ok": bool(detect_ok),
    }

    return np.array(boxes[:EXPECTED_TOTAL_BOXES], dtype=np.int32), detect_ok, meta


# =============================================================================
# Fallback template
# =============================================================================

def compute_fallback_template(
    src_dir: Path,
    max_images: Optional[int] = None,
) -> np.ndarray:
    """
    Build a 4-box fallback template using the median of successful detections.

    Returns:
        template_boxes: (4, 4) int32 array [x, y, w, h]
    """
    img_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"):
        img_paths.extend(sorted(src_dir.glob(ext)))

    if max_images is not None:
        img_paths = img_paths[:max_images]

    good_boxes = []

    for img_path in tqdm(img_paths, desc="Computing fallback template"):
        gray = np.array(Image.open(img_path).convert("L"))
        boxes, detect_ok, _ = detect_roi_boxes(gray)
        if detect_ok and np.all(boxes[:, 2] > 0) and np.all(boxes[:, 3] > 0):
            good_boxes.append(boxes)

    if len(good_boxes) == 0:
        raise RuntimeError(
            "Could not compute fallback template because no successful detections were found."
        )

    good_boxes = np.stack(good_boxes, axis=0)  # (N, 4, 4)
    template = np.median(good_boxes, axis=0)
    template = np.round(template).astype(np.int32)

    FALLBACK_TEMPLATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(FALLBACK_TEMPLATE_PATH, template)

    return template


def load_or_build_fallback_template(template_source_dir: Path) -> np.ndarray:
    if FALLBACK_TEMPLATE_PATH.exists():
        return np.load(FALLBACK_TEMPLATE_PATH)
    return compute_fallback_template(template_source_dir)


# =============================================================================
# ROI extraction / saving
# =============================================================================

def crop_and_resize(gray: np.ndarray, box: np.ndarray, roi_size: int = ROI_SIZE) -> np.ndarray:
    x, y, w, h = [int(v) for v in box]

    if w <= 0 or h <= 0:
        return np.zeros((roi_size, roi_size), dtype=np.uint8)

    crop = gray[y:y + h, x:x + w]
    if crop.size == 0:
        return np.zeros((roi_size, roi_size), dtype=np.uint8)

    crop = cv2.resize(crop, (roi_size, roi_size), interpolation=cv2.INTER_AREA)
    return crop.astype(np.uint8)


def make_4channel_roi(gray: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    rois = [crop_and_resize(gray, boxes[i], roi_size=ROI_SIZE) for i in range(NUM_ROIS)]
    arr = np.stack(rois, axis=0)  # (4, H, W)
    return arr.astype(np.uint8)


def make_mosaic(arr_4chw: np.ndarray) -> np.ndarray:
    """
    Build a 2x2 visualization image from the 4 channels.
    Order:
        0 2
        1 3
    """
    r1 = np.concatenate([arr_4chw[0], arr_4chw[2]], axis=1)
    r2 = np.concatenate([arr_4chw[1], arr_4chw[3]], axis=1)
    return np.concatenate([r1, r2], axis=0).astype(np.uint8)


def draw_boxes_on_original(
    gray: np.ndarray,
    boxes: np.ndarray,
    roi_mode: str,
) -> np.ndarray:
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    color = (0, 255, 0) if roi_mode == "detected" else (0, 165, 255)

    for i, (x, y, w, h) in enumerate(boxes):
        x, y, w, h = int(x), int(y), int(w), int(h)
        if w <= 0 or h <= 0:
            continue

        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
        label = f"ROI{i+1}"
        cv2.putText(
            vis,
            label,
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        vis,
        f"mode={roi_mode}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        color,
        2,
        cv2.LINE_AA,
    )

    return vis


def process_one_image(
    img_path: Path,
    out_dir: Path,
    debug_dir: Path,
    fallback_template: np.ndarray,
    save_debug: bool = True,
):
    gray = np.array(Image.open(img_path).convert("L"))

    boxes, detect_ok, meta = detect_roi_boxes(gray)

    if detect_ok and np.all(boxes[:, 2] > 0) and np.all(boxes[:, 3] > 0):
        final_boxes = boxes
        roi_mode = "detected"
    else:
        final_boxes = fallback_template.copy()
        roi_mode = "fallback"

    roi_4ch = make_4channel_roi(gray, final_boxes)

    stem = img_path.stem
    np.save(out_dir / f"{stem}.npy", roi_4ch)

    metadata = {
        "image_id": img_path.name,
        "stem": stem,
        "roi_mode": roi_mode,
        "detect_ok": bool(detect_ok),
        "boxes": final_boxes.tolist(),
        "detector_meta": meta,
        "roi_shape": list(roi_4ch.shape),
    }

    with open(out_dir / f"{stem}.json", "w") as f:
        json.dump(metadata, f, indent=2)

    if save_debug:
        mosaic = make_mosaic(roi_4ch)
        Image.fromarray(mosaic).save(debug_dir / f"{stem}_mosaic.png")

        boxed = draw_boxes_on_original(gray, final_boxes, roi_mode=roi_mode)
        Image.fromarray(boxed).save(debug_dir / f"{stem}_boxed.png")

    return {
        "image_id": img_path.name,
        "roi_mode": roi_mode,
        "detect_ok": bool(detect_ok),
        "left_confidence": meta["left_confidence"],
        "right_confidence": meta["right_confidence"],
        "total_confidence": meta["total_confidence"],
    }


def _worker(args):
    img_path, out_dir, debug_dir, fallback_template, save_debug = args
    img_path = Path(img_path)
    try:
        result = process_one_image(
            img_path=img_path,
            out_dir=out_dir,
            debug_dir=debug_dir,
            fallback_template=fallback_template,
            save_debug=save_debug,
        )
        return True, result, None
    except Exception as e:
        return False, {"image_id": img_path.name}, str(e)


# =============================================================================
# Split processing
# =============================================================================

def collect_images(src_dir: Path) -> list[Path]:
    imgs = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"):
        imgs.extend(sorted(src_dir.glob(ext)))
    return sorted(imgs)


def process_split(
    src_dir: Path,
    out_dir: Path,
    split_name: str,
    fallback_template: np.ndarray,
    workers: int = 8,
    save_debug: bool = True,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = ROI_BOX_DEBUG_DIR / split_name
    debug_dir.mkdir(parents=True, exist_ok=True)

    img_paths = collect_images(src_dir)
    print(f"[{split_name}] Found {len(img_paths)} images in {src_dir}")

    if len(img_paths) == 0:
        print(f"[{split_name}] No images found. Skipping.")
        return

    args_list = [
        (str(p), out_dir, debug_dir, fallback_template, save_debug)
        for p in img_paths
    ]

    results = []
    failures = []

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_worker, a): a[0] for a in args_list}
        with tqdm(total=len(img_paths), desc=f"Building ROI set: {split_name}") as pbar:
            for fut in as_completed(futs):
                ok, result, err = fut.result()
                if ok:
                    results.append(result)
                else:
                    failures.append((result["image_id"], err))
                pbar.update(1)

    # Save manifest
    manifest_path = out_dir / "_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(
            {
                "split": split_name,
                "num_images": len(img_paths),
                "num_success": len(results),
                "num_failures": len(failures),
                "results": results,
                "failures": failures,
            },
            f,
            indent=2,
        )

    print(f"[{split_name}] Done: {len(results)}/{len(img_paths)} succeeded")
    if failures:
        print(f"[{split_name}] Failures: {len(failures)}")
        fail_log = out_dir / "_failures.txt"
        fail_log.write_text("\n".join([f"{name}\t{err}" for name, err in failures]))


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--save_debug", action="store_true")
    parser.add_argument("--rebuild_fallback", action="store_true")
    parser.add_argument(
        "--template_source",
        type=Path,
        default=TRAIN_DIR,
        help="Directory used to compute fallback template if needed.",
    )
    parser.add_argument(
        "--only_split",
        type=str,
        default=None,
        choices=[None, "train", "validation", "test", "unlabeled"],
        help="Process only one split.",
    )
    args = parser.parse_args()

    ensure_project_dirs()

    if args.rebuild_fallback and FALLBACK_TEMPLATE_PATH.exists():
        FALLBACK_TEMPLATE_PATH.unlink()

    fallback_template = load_or_build_fallback_template(args.template_source)
    print(f"Fallback template loaded: {FALLBACK_TEMPLATE_PATH}")

    split_map = {
        "train": (TRAIN_DIR, ROI_TRAIN_DIR),
        "validation": (VAL_DIR, ROI_VAL_DIR),
        "test": (TEST_DIR, ROI_TEST_DIR),
        "unlabeled": (UNLABELED_DIR, ROI_UNLABELED_DIR),
    }

    if args.only_split is not None:
        src_dir, out_dir = split_map[args.only_split]
        process_split(
            src_dir=src_dir,
            out_dir=out_dir,
            split_name=args.only_split,
            fallback_template=fallback_template,
            workers=args.workers,
            save_debug=args.save_debug,
        )
    else:
        for split_name, (src_dir, out_dir) in split_map.items():
            process_split(
                src_dir=src_dir,
                out_dir=out_dir,
                split_name=split_name,
                fallback_template=fallback_template,
                workers=args.workers,
                save_debug=args.save_debug,
            )


if __name__ == "__main__":
    main()