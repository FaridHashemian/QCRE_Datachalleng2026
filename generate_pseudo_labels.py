from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18
from tqdm import tqdm


# ============================================================
# Paths
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "DataFile"
ROI_UNLABELED_SAMPLE_DIR = DATA_ROOT / "roi_data" / "unlabeled_sample"
WEIGHTS_DIR = DATA_ROOT / "trained_weights"
LOG_DIR = DATA_ROOT / "logs"
PSEUDO_DIR = DATA_ROOT / "pseudo_labels"


# ============================================================
# Model (must match train.py)
# ============================================================
class ResNet18Backbone4Ch(nn.Module):
    def __init__(self):
        super().__init__()
        model = resnet18(weights=None)

        old_conv = model.conv1
        model.conv1 = nn.Conv2d(
            in_channels=4,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        self.encoder = nn.Sequential(*list(model.children())[:-1])
        self.out_dim = 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        z = torch.flatten(z, 1)
        return z


class MultiTaskClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, feat_dim: int = 512, dropout: float = 0.2):
        super().__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(dropout)
        self.bin_head = nn.Linear(feat_dim, 1)
        self.dt_head = nn.Linear(feat_dim, 3)

    def forward(self, x: torch.Tensor):
        feat = self.backbone(x)
        feat = self.dropout(feat)
        bin_logit = self.bin_head(feat)
        dt_logit = self.dt_head(feat)
        return bin_logit, dt_logit


# ============================================================
# Dataset
# ============================================================
class UnlabeledROITensorDataset(Dataset):
    def __init__(self, roi_dir: Path):
        self.roi_dir = roi_dir
        self.files = sorted(roi_dir.glob("*.npy"))
        if len(self.files) == 0:
            raise RuntimeError(f"No .npy files found in {roi_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        arr = np.load(path)
        if arr.ndim != 3 or arr.shape[0] != 4:
            raise ValueError(f"Unexpected ROI tensor shape {arr.shape} for {path}")
        x = torch.from_numpy(arr).float() / 255.0
        image_id = f"{path.stem}.jpg"
        return x, image_id, path.stem


# ============================================================
# Utils
# ============================================================
def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def choose_checkpoint(run_name: str | None, checkpoint_path: str | None) -> Path:
    if checkpoint_path is not None:
        ckpt = Path(checkpoint_path)
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        return ckpt

    if run_name is None:
        raise ValueError("Provide either --checkpoint_path or --run_name")

    candidates = [
        LOG_DIR / run_name / "supervised_best_train_proxy.pt",
        WEIGHTS_DIR / f"{run_name}_supervised_best_train_proxy.pt",
        LOG_DIR / run_name / "supervised_last.pt",
    ]
    for c in candidates:
        if c.exists():
            return c

    raise FileNotFoundError(f"Could not find supervised checkpoint for run_name={run_name}")


def load_model_from_checkpoint(ckpt_path: Path, device: torch.device) -> MultiTaskClassifier:
    ckpt = torch.load(ckpt_path, map_location=device)

    dropout = 0.2
    if "config" in ckpt and isinstance(ckpt["config"], dict) and "dropout" in ckpt["config"]:
        dropout = float(ckpt["config"]["dropout"])

    backbone = ResNet18Backbone4Ch()
    model = MultiTaskClassifier(backbone=backbone, feat_dim=backbone.out_dim, dropout=dropout)

    if "model_state_dict" not in ckpt:
        raise ValueError("Checkpoint does not contain model_state_dict. Use supervised checkpoint.")

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


# ============================================================
# Pseudo-label generation
# ============================================================
@torch.no_grad()
def generate_pseudo_labels(
    model: MultiTaskClassifier,
    loader: DataLoader,
    device: torch.device,
    defect_pos_thresh: float,
    defect_neg_thresh: float,
    dt_pos_thresh: float,
    dt_neg_thresh: float,
    mode: str,
):
    rows = []

    for x, image_id, stem in tqdm(loader, desc="Generating pseudo labels"):
        x = x.to(device, non_blocking=True)

        bin_logit, dt_logit = model(x)
        defect_prob = torch.sigmoid(bin_logit).squeeze(1).cpu().numpy()
        dt_prob = torch.sigmoid(dt_logit).cpu().numpy()

        for i in range(len(image_id)):
            p_def = float(defect_prob[i])
            p_dt1 = float(dt_prob[i, 0])
            p_dt2 = float(dt_prob[i, 1])
            p_dt3 = float(dt_prob[i, 2])

            keep = False
            defect = None
            dt1 = None
            dt2 = None
            dt3 = None
            label_mode = None

            # confident non-defect
            if p_def <= defect_neg_thresh:
                keep = True
                defect = 0
                dt1, dt2, dt3 = 0, 0, 0
                label_mode = "confident_negative"

            # confident defect
            elif p_def >= defect_pos_thresh:
                if mode == "binary_only":
                    keep = True
                    defect = 1
                    # leave DTs missing; they will be ignored during DT pseudo loss
                    dt1, dt2, dt3 = -1, -1, -1
                    label_mode = "confident_positive_binary_only"
                else:
                    dt_vals = []
                    valid = True
                    for p in [p_dt1, p_dt2, p_dt3]:
                        if p >= dt_pos_thresh:
                            dt_vals.append(1)
                        elif p <= dt_neg_thresh:
                            dt_vals.append(0)
                        else:
                            valid = False
                            break

                    if valid:
                        keep = True
                        defect = 1
                        dt1, dt2, dt3 = dt_vals
                        label_mode = "confident_positive_with_dt"

            if keep:
                rows.append({
                    "Image_id": image_id[i],
                    "stem": stem[i],
                    "Defect": int(defect),
                    "DT1": int(dt1),
                    "DT2": int(dt2),
                    "DT3": int(dt3),
                    "pseudo_mode": label_mode,
                    "Defect_prob": p_def,
                    "DT1_prob": p_dt1,
                    "DT2_prob": p_dt2,
                    "DT3_prob": p_dt3,
                })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--roi_unlabeled_dir", type=str, default=str(ROI_UNLABELED_SAMPLE_DIR))
    parser.add_argument("--output_csv", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--defect_pos_thresh", type=float, default=0.97)
    parser.add_argument("--defect_neg_thresh", type=float, default=0.03)
    parser.add_argument("--dt_pos_thresh", type=float, default=0.95)
    parser.add_argument("--dt_neg_thresh", type=float, default=0.05)
    parser.add_argument("--mode", type=str, default="binary_only", choices=["binary_only", "binary_and_dt"])

    args = parser.parse_args()

    PSEUDO_DIR.mkdir(parents=True, exist_ok=True)

    if args.output_csv is None:
        tag = args.run_name if args.run_name is not None else Path(args.checkpoint_path).stem
        args.output_csv = str(PSEUDO_DIR / f"{tag}_pseudo_labels_{args.mode}.csv")

    device = resolve_device(args.device)
    ckpt_path = choose_checkpoint(args.run_name, args.checkpoint_path)

    print(f"Using device: {device}")
    print(f"Using checkpoint: {ckpt_path}")

    dataset = UnlabeledROITensorDataset(Path(args.roi_unlabeled_dir))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    model = load_model_from_checkpoint(ckpt_path, device)
    df = generate_pseudo_labels(
        model=model,
        loader=loader,
        device=device,
        defect_pos_thresh=args.defect_pos_thresh,
        defect_neg_thresh=args.defect_neg_thresh,
        dt_pos_thresh=args.dt_pos_thresh,
        dt_neg_thresh=args.dt_neg_thresh,
        mode=args.mode,
    )

    df.to_csv(args.output_csv, index=False)
    print(f"Saved pseudo labels to: {args.output_csv}")
    print(f"Num pseudo-labeled samples: {len(df)}")

    if len(df) > 0:
        print(df["pseudo_mode"].value_counts(dropna=False))


if __name__ == "__main__":
    main()