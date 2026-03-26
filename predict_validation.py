from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18
from tqdm import tqdm


# ============================================================
# Defaults
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "DataFile"

ROI_VAL_DIR = DATA_ROOT / "roi_data" / "validation"
WEIGHTS_DIR = DATA_ROOT / "trained_weights"
LOG_DIR = DATA_ROOT / "logs"

DEFAULT_OUTPUT_CSV = DATA_ROOT / "validation_predictions.csv"


# ============================================================
# Model definition (must match train.py)
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

class ValidationROITensorDataset(Dataset):
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
# Utilities
# ============================================================

def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def infer_dropout_from_checkpoint(ckpt: dict, default: float = 0.2) -> float:
    cfg = ckpt.get("config", {})
    if isinstance(cfg, dict) and "dropout" in cfg:
        return float(cfg["dropout"])
    return default


def load_model_from_checkpoint(ckpt_path: Path, device: torch.device) -> MultiTaskClassifier:
    ckpt = torch.load(ckpt_path, map_location=device)

    dropout = infer_dropout_from_checkpoint(ckpt, default=0.2)

    backbone = ResNet18Backbone4Ch()
    model = MultiTaskClassifier(
        backbone=backbone,
        feat_dim=backbone.out_dim,
        dropout=dropout,
    )

    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        raise ValueError(
            f"Checkpoint {ckpt_path} does not contain 'model_state_dict'. "
            "Use the supervised checkpoint, not the SSL backbone checkpoint."
        )

    model.to(device)
    model.eval()
    return model


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

    raise FileNotFoundError(
        f"Could not find a supervised checkpoint for run_name={run_name}. "
        f"Tried: {[str(c) for c in candidates]}"
    )


# ============================================================
# Prediction
# ============================================================

@torch.no_grad()
def predict(
    model: MultiTaskClassifier,
    loader: DataLoader,
    device: torch.device,
    bin_threshold: float = 0.5,
    dt_threshold: float = 0.5,
):
    rows = []

    for x, image_ids, stems in tqdm(loader, desc="Predicting validation"):
        x = x.to(device, non_blocking=True)

        bin_logit, dt_logit = model(x)

        bin_prob = torch.sigmoid(bin_logit).squeeze(1).cpu().numpy()
        dt_prob = torch.sigmoid(dt_logit).cpu().numpy()

        bin_pred = (bin_prob >= bin_threshold).astype(np.int64)
        dt_pred = (dt_prob >= dt_threshold).astype(np.int64)

        # Optional consistency rule:
        # if Defect=0, force all DTs to 0
        for i in range(len(image_ids)):
            defect = int(bin_pred[i])
            dt1, dt2, dt3 = [int(v) for v in dt_pred[i]]

            if defect == 0:
                dt1, dt2, dt3 = 0, 0, 0

            rows.append({
                "Image_id": image_ids[i],
                "Defect": defect,
                "DT1(Missing_Perforations)": dt1,
                "DT2(Touching_Perforations)": dt2,
                "DT3(Out_Of_Bounds)": dt3,
                "_Defect_prob": float(bin_prob[i]),
                "_DT1_prob": float(dt_prob[i, 0]),
                "_DT2_prob": float(dt_prob[i, 1]),
                "_DT3_prob": float(dt_prob[i, 2]),
            })

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--roi_val_dir", type=str, default=str(ROI_VAL_DIR))
    parser.add_argument("--output_csv", type=str, default=str(DEFAULT_OUTPUT_CSV))
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--bin_threshold", type=float, default=0.5)
    parser.add_argument("--dt_threshold", type=float, default=0.5)
    parser.add_argument("--save_probs", action="store_true")
    args = parser.parse_args()

    device = resolve_device(args.device)
    ckpt_path = choose_checkpoint(args.run_name, args.checkpoint_path)

    print(f"Using device: {device}")
    print(f"Using checkpoint: {ckpt_path}")

    dataset = ValidationROITensorDataset(Path(args.roi_val_dir))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    model = load_model_from_checkpoint(ckpt_path, device)
    rows = predict(
        model=model,
        loader=loader,
        device=device,
        bin_threshold=args.bin_threshold,
        dt_threshold=args.dt_threshold,
    )

    df = pd.DataFrame(rows)

    # Keep exact requested output header
    output_cols = [
        "Image_id",
        "Defect",
        "DT1(Missing_Perforations)",
        "DT2(Touching_Perforations)",
        "DT3(Out_Of_Bounds)",
    ]

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df[output_cols].to_csv(out_path, index=False)

    print(f"Saved validation predictions to: {out_path}")

    if args.save_probs:
        prob_path = out_path.with_name(out_path.stem + "_with_probs.csv")
        prob_cols = output_cols + [
            "_Defect_prob",
            "_DT1_prob",
            "_DT2_prob",
            "_DT3_prob",
        ]
        df[prob_cols].to_csv(prob_path, index=False)
        print(f"Saved probability file to: {prob_path}")


if __name__ == "__main__":
    main()