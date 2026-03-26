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

ROI_TEST_DIR = DATA_ROOT / "roi_data" / "test"
TEMPLATE_CSV = DATA_ROOT / "test_detection.csv"
DEFAULT_OUTPUT_CSV = DATA_ROOT / "submission_test.csv"

WEIGHTS_DIR = DATA_ROOT / "trained_weights"
LOG_DIR = DATA_ROOT / "logs"


# ============================================================
# Model
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
        return torch.flatten(z, 1)


class MultiTaskClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, feat_dim: int = 512, dropout: float = 0.2):
        super().__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(dropout)
        self.bin_head = nn.Linear(feat_dim, 1)
        self.dt_head = nn.Linear(feat_dim, 3)

    def forward(self, x: torch.Tensor):
        feat = self.dropout(self.backbone(x))
        return self.bin_head(feat), self.dt_head(feat)


# ============================================================
# Dataset
# ============================================================
class TestDataset(Dataset):
    def __init__(self, roi_dir: Path):
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
        return x, image_id


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


def choose_checkpoint(run_name: str | None, checkpoint_path: str | None) -> Path:
    if checkpoint_path is not None:
        ckpt = Path(checkpoint_path)
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        return ckpt

    if run_name is None:
        raise ValueError("Provide either --run_name or --checkpoint_path")

    candidates = [
        LOG_DIR / run_name / "supervised_best_train_proxy.pt",
        WEIGHTS_DIR / f"{run_name}_supervised_best_train_proxy.pt",
        LOG_DIR / run_name / "supervised_last.pt",
        LOG_DIR / run_name / "pseudo_best_train_proxy.pt",
        WEIGHTS_DIR / f"{run_name}_pseudo_best_train_proxy.pt",
        LOG_DIR / run_name / "pseudo_last.pt",
        LOG_DIR / run_name / f"{run_name}_pseudo_best_train_proxy.pt",
        WEIGHTS_DIR / f"{run_name}_pseudo_best_train_proxy.pt",
    ]

    for c in candidates:
        if c.exists():
            return c

    tried = "\n".join(str(c) for c in candidates)
    raise FileNotFoundError(f"Could not find checkpoint for run_name={run_name}. Tried:\n{tried}")


def load_model(ckpt_path: Path, device: torch.device) -> MultiTaskClassifier:
    ckpt = torch.load(ckpt_path, map_location=device)

    dropout = infer_dropout_from_checkpoint(ckpt, default=0.2)

    backbone = ResNet18Backbone4Ch()
    model = MultiTaskClassifier(backbone=backbone, feat_dim=backbone.out_dim, dropout=dropout)

    if "model_state_dict" not in ckpt:
        raise ValueError(f"Checkpoint {ckpt_path} does not contain model_state_dict")

    state = ckpt["model_state_dict"]
    # In case the checkpoint was saved from DataParallel/DDP
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


# ============================================================
# Inference
# ============================================================
@torch.no_grad()
def run_inference(
    model: MultiTaskClassifier,
    loader: DataLoader,
    device: torch.device,
    defect_threshold: float,
    dt1_threshold: float,
    dt2_threshold: float,
    dt3_threshold: float,
):
    results = {}

    for x, image_ids in tqdm(loader, desc="Predicting test"):
        x = x.to(device, non_blocking=True)

        bin_logit, dt_logit = model(x)

        p_def = torch.sigmoid(bin_logit).squeeze(1).cpu().numpy()
        p_dt = torch.sigmoid(dt_logit).cpu().numpy()

        for i, img_id in enumerate(image_ids):
            defect = int(p_def[i] >= defect_threshold)
            dt1 = int(p_dt[i, 0] >= dt1_threshold)
            dt2 = int(p_dt[i, 1] >= dt2_threshold)
            dt3 = int(p_dt[i, 2] >= dt3_threshold)

            if defect == 0:
                dt1, dt2, dt3 = 0, 0, 0

            results[img_id] = {
                "Defect": defect,
                "DT1(Missing_Perforations)": dt1,
                "DT2(Touching_Perforations)": dt2,
                "DT3(Out_Of_Bounds)": dt3,
                "_Defect_prob": float(p_def[i]),
                "_DT1_prob": float(p_dt[i, 0]),
                "_DT2_prob": float(p_dt[i, 1]),
                "_DT3_prob": float(p_dt[i, 2]),
            }

    return results


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--roi_test_dir", type=str, default=str(ROI_TEST_DIR))
    parser.add_argument("--template_csv", type=str, default=str(TEMPLATE_CSV))
    parser.add_argument("--output_csv", type=str, default=str(DEFAULT_OUTPUT_CSV))
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--defect_threshold", type=float, default=0.5)
    parser.add_argument("--dt1_threshold", type=float, default=0.5)
    parser.add_argument("--dt2_threshold", type=float, default=0.5)
    parser.add_argument("--dt3_threshold", type=float, default=0.5)

    parser.add_argument("--save_probs", action="store_true")
    args = parser.parse_args()

    device = resolve_device(args.device)
    ckpt_path = choose_checkpoint(args.run_name, args.checkpoint_path)

    print(f"Using device: {device}")
    print(f"Using checkpoint: {ckpt_path}")

    dataset = TestDataset(Path(args.roi_test_dir))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    model = load_model(ckpt_path, device)

    preds = run_inference(
        model=model,
        loader=loader,
        device=device,
        defect_threshold=args.defect_threshold,
        dt1_threshold=args.dt1_threshold,
        dt2_threshold=args.dt2_threshold,
        dt3_threshold=args.dt3_threshold,
    )

    df = pd.read_csv(args.template_csv)

    required_cols = [
        "Image_id",
        "Defect",
        "DT1(Missing_Perforations)",
        "DT2(Touching_Perforations)",
        "DT3(Out_Of_Bounds)",
    ]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column in template: {c}")

    missing_ids = []
    for i in range(len(df)):
        img_id = df.loc[i, "Image_id"]
        if img_id not in preds:
            missing_ids.append(img_id)
            continue

        row = preds[img_id]
        df.loc[i, "Defect"] = int(row["Defect"])
        df.loc[i, "DT1(Missing_Perforations)"] = int(row["DT1(Missing_Perforations)"])
        df.loc[i, "DT2(Touching_Perforations)"] = int(row["DT2(Touching_Perforations)"])
        df.loc[i, "DT3(Out_Of_Bounds)"] = int(row["DT3(Out_Of_Bounds)"])

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df[required_cols].to_csv(out_path, index=False)
    print(f"Saved submission CSV to: {out_path}")

    if missing_ids:
        miss_path = out_path.with_name(out_path.stem + "_missing_ids.txt")
        miss_path.write_text("\n".join(missing_ids))
        print(f"Warning: {len(missing_ids)} template IDs had no matching .npy file. Saved list to: {miss_path}")

    if args.save_probs:
        prob_rows = []
        for img_id in df["Image_id"].tolist():
            if img_id in preds:
                r = preds[img_id]
                prob_rows.append({
                    "Image_id": img_id,
                    "_Defect_prob": r["_Defect_prob"],
                    "_DT1_prob": r["_DT1_prob"],
                    "_DT2_prob": r["_DT2_prob"],
                    "_DT3_prob": r["_DT3_prob"],
                })
        prob_df = pd.DataFrame(prob_rows)
        prob_path = out_path.with_name(out_path.stem + "_with_probs.csv")
        prob_df.to_csv(prob_path, index=False)
        print(f"Saved probability CSV to: {prob_path}")


if __name__ == "__main__":
    main()
