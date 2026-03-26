from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def make_mosaic(arr_4chw: np.ndarray) -> np.ndarray:
    """
    Input shape: (4, H, W)
    Layout:
        ROI1 ROI3
        ROI2 ROI4
    """
    if arr_4chw.ndim != 3 or arr_4chw.shape[0] != 4:
        raise ValueError(f"Expected shape (4,H,W), got {arr_4chw.shape}")

    r1 = np.concatenate([arr_4chw[0], arr_4chw[2]], axis=1)
    r2 = np.concatenate([arr_4chw[1], arr_4chw[3]], axis=1)
    mosaic = np.concatenate([r1, r2], axis=0)
    return mosaic.astype(np.uint8)


def add_labels_to_mosaic(mosaic: np.ndarray) -> Image.Image:
    img = Image.fromarray(mosaic).convert("RGB")
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), "ROI1", fill=(255, 0, 0))
    draw.text((mosaic.shape[1] // 2 + 10, 10), "ROI3", fill=(255, 0, 0))
    draw.text((10, mosaic.shape[0] // 2 + 10), "ROI2", fill=(255, 0, 0))
    draw.text((mosaic.shape[1] // 2 + 10, mosaic.shape[0] // 2 + 10), "ROI4", fill=(255, 0, 0))
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_dir",
        type=Path,
        default=Path(__file__).resolve().parent / "DataFile" / "roi_data" / "unlabeled_sample",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path(__file__).resolve().parent / "DataFile" / "logs" / "debug_extracted_samples",
    )
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(args.src_dir.glob("*.npy"))
    if len(files) == 0:
        raise RuntimeError(f"No .npy files found in {args.src_dir}")

    rng = random.Random(args.seed)
    chosen = files if len(files) <= args.num_samples else rng.sample(files, args.num_samples)

    manifest_lines = [
        f"src_dir: {args.src_dir}",
        f"out_dir: {args.out_dir}",
        f"num_available: {len(files)}",
        f"num_selected: {len(chosen)}",
        f"seed: {args.seed}",
        "",
    ]

    for npy_path in chosen:
        arr = np.load(npy_path)
        if arr.ndim != 3 or arr.shape[0] != 4:
            print(f"Skipping invalid tensor shape for {npy_path.name}: {arr.shape}")
            continue

        mosaic = make_mosaic(arr)
        img = add_labels_to_mosaic(mosaic)
        img.save(args.out_dir / f"{npy_path.stem}_mosaic.png")
        manifest_lines.append(npy_path.name)

    (args.out_dir / "_manifest.txt").write_text("\n".join(manifest_lines))
    print(f"Saved {len(chosen)} debug mosaics to {args.out_dir}")


if __name__ == "__main__":
    main()