from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18
from tqdm import tqdm


# ============================================================
# Paths
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "DataFile"

LOG_DIR = DATA_ROOT / "logs"
WEIGHTS_DIR = DATA_ROOT / "trained_weights"
TRAIN_LABELS_CSV = DATA_ROOT / "train_labels.csv"
ROI_TRAIN_DIR = DATA_ROOT / "roi_data" / "train"
ROI_UNLABELED_SAMPLE_DIR = DATA_ROOT / "roi_data" / "unlabeled_sample"
PSEUDO_DIR = DATA_ROOT / "pseudo_labels"


# ============================================================
# Utilities
# ============================================================
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def now_str() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def get_num_workers(requested: int) -> int:
    if requested >= 0:
        return requested
    cpu_count = os.cpu_count() or 4
    return max(2, min(8, cpu_count // 2))


# ============================================================
# Logging
# ============================================================
class JsonlLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, payload: Dict):
        with self.path.open("a") as f:
            f.write(json.dumps(payload) + "\n")


def setup_run_logger(run_dir: Path) -> logging.Logger:
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"qcre_pseudo_{run_dir.name}")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(run_dir / "train.log")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


# ============================================================
# CSV parsing
# ============================================================
def _pick_existing_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def load_train_labels(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    image_col = _pick_existing_col(df, ["image_id", "Image_id"])
    defect_col = _pick_existing_col(df, ["defect", "Defect"])
    dt1_col = _pick_existing_col(df, ["DT1", "dt1", "DT1_MP", "dt1_mp"])
    dt2_col = _pick_existing_col(df, ["DT2", "dt2", "DT2_TP", "dt2_tp"])
    dt3_col = _pick_existing_col(df, ["DT3", "dt3", "DT3_OOB", "dt3_oob"])

    missing = []
    for name, col in [
        ("image_id", image_col),
        ("defect", defect_col),
        ("DT1", dt1_col),
        ("DT2", dt2_col),
        ("DT3", dt3_col),
    ]:
        if col is None:
            missing.append(name)

    if missing:
        raise ValueError(
            f"Could not infer required columns from {csv_path}. Missing: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    out = pd.DataFrame({
        "image_id": df[image_col].astype(str),
        "defect": df[defect_col].astype(int),
        "DT1": df[dt1_col].astype(int),
        "DT2": df[dt2_col].astype(int),
        "DT3": df[dt3_col].astype(int),
    })
    out["stem"] = out["image_id"].map(lambda x: Path(x).stem)
    out["is_pseudo"] = 0
    return out


def load_pseudo_labels(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required = ["Image_id", "Defect", "DT1", "DT2", "DT3"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Pseudo-label CSV missing columns: {missing}")

    out = pd.DataFrame({
        "image_id": df["Image_id"].astype(str),
        "defect": df["Defect"].astype(int),
        "DT1": df["DT1"].astype(int),
        "DT2": df["DT2"].astype(int),
        "DT3": df["DT3"].astype(int),
    })
    out["stem"] = out["image_id"].map(lambda x: Path(x).stem)
    out["is_pseudo"] = 1
    return out


# ============================================================
# Augmentations
# ============================================================
class ROITrainAugment:
    def __init__(
        self,
        p_hflip: float = 0.5,
        p_vflip: float = 0.2,
        brightness: float = 0.15,
        contrast: float = 0.15,
        noise_std: float = 0.02,
    ):
        self.p_hflip = p_hflip
        self.p_vflip = p_vflip
        self.brightness = brightness
        self.contrast = contrast
        self.noise_std = noise_std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.p_hflip:
            x = torch.flip(x, dims=[2])
        if torch.rand(1).item() < self.p_vflip:
            x = torch.flip(x, dims=[1])

        b = 1.0 + (torch.rand(1).item() * 2.0 - 1.0) * self.brightness
        x = x * b

        mean = x.mean(dim=(1, 2), keepdim=True)
        c = 1.0 + (torch.rand(1).item() * 2.0 - 1.0) * self.contrast
        x = (x - mean) * c + mean

        x = x + torch.randn_like(x) * self.noise_std
        return torch.clamp(x, 0.0, 1.0)


class ROIIdentity:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x


# ============================================================
# Datasets
# ============================================================
class MixedROITensorDataset(Dataset):
    def __init__(
        self,
        real_labels_df: pd.DataFrame,
        pseudo_labels_df: pd.DataFrame,
        roi_train_dir: Path,
        roi_unlabeled_dir: Path,
        transform=None,
        failure_logger: Optional[JsonlLogger] = None,
    ):
        self.transform = transform if transform is not None else ROIIdentity()
        self.failure_logger = failure_logger

        records = []

        for _, row in real_labels_df.iterrows():
            stem = row["stem"]
            npy_path = roi_train_dir / f"{stem}.npy"
            if npy_path.exists():
                records.append({
                    "stem": stem,
                    "path": npy_path,
                    "defect": int(row["defect"]),
                    "dt": np.array([int(row["DT1"]), int(row["DT2"]), int(row["DT3"])], dtype=np.float32),
                    "is_pseudo": 0,
                })

        for _, row in pseudo_labels_df.iterrows():
            stem = row["stem"]
            npy_path = roi_unlabeled_dir / f"{stem}.npy"
            if npy_path.exists():
                records.append({
                    "stem": stem,
                    "path": npy_path,
                    "defect": int(row["defect"]),
                    "dt": np.array([int(row["DT1"]), int(row["DT2"]), int(row["DT3"])], dtype=np.float32),
                    "is_pseudo": 1,
                })

        self.records = records
        if len(self.records) == 0:
            raise RuntimeError("No records found for mixed dataset.")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        try:
            arr = np.load(rec["path"])
            if arr.ndim != 3 or arr.shape[0] != 4:
                raise ValueError(f"Unexpected ROI tensor shape {arr.shape}")

            x = torch.from_numpy(arr).float() / 255.0
            x = self.transform(x)

            y_bin = torch.tensor([rec["defect"]], dtype=torch.float32)
            y_dt = torch.tensor(rec["dt"], dtype=torch.float32)
            is_pseudo = torch.tensor([rec["is_pseudo"]], dtype=torch.float32)

            return x, y_bin, y_dt, is_pseudo, rec["stem"]
        except Exception as e:
            if self.failure_logger is not None:
                self.failure_logger.write({
                    "stage": "mixed_dataset",
                    "file": str(rec["path"]),
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                })
            raise


class RealOnlyEvalDataset(Dataset):
    def __init__(self, labels_df: pd.DataFrame, roi_train_dir: Path):
        self.records = []
        for _, row in labels_df.iterrows():
            stem = row["stem"]
            npy_path = roi_train_dir / f"{stem}.npy"
            if npy_path.exists():
                self.records.append({
                    "stem": stem,
                    "path": npy_path,
                    "defect": int(row["defect"]),
                    "dt": np.array([int(row["DT1"]), int(row["DT2"]), int(row["DT3"])], dtype=np.float32),
                })

        if len(self.records) == 0:
            raise RuntimeError("No real eval records found.")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        arr = np.load(rec["path"])
        x = torch.from_numpy(arr).float() / 255.0
        y_bin = torch.tensor([rec["defect"]], dtype=torch.float32)
        y_dt = torch.tensor(rec["dt"], dtype=torch.float32)
        return x, y_bin, y_dt, rec["stem"]


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

    def forward(self, x):
        z = self.encoder(x)
        return torch.flatten(z, 1)


class MultiTaskClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, feat_dim: int = 512, dropout: float = 0.2):
        super().__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(dropout)
        self.bin_head = nn.Linear(feat_dim, 1)
        self.dt_head = nn.Linear(feat_dim, 3)

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.dropout(feat)
        return self.bin_head(feat), self.dt_head(feat)


# ============================================================
# Metrics
# ============================================================
def micro_f1_binary(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.astype(np.int64).reshape(-1)
    y_pred = y_pred.astype(np.int64).reshape(-1)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    denom = 2 * tp + fp + fn
    if denom == 0:
        return 1.0
    return (2.0 * tp) / denom


def micro_f1_multilabel(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.astype(np.int64).reshape(-1)
    y_pred = y_pred.astype(np.int64).reshape(-1)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    denom = 2 * tp + fp + fn
    if denom == 0:
        return 1.0
    return (2.0 * tp) / denom


def exact_match_accuracy_multilabel(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).all(axis=1).mean())


def challenge_proxy_score(y_bin_true, y_bin_pred, y_dt_true, y_dt_pred):
    bin_acc = float((y_bin_true == y_bin_pred).mean())
    bin_f1 = micro_f1_binary(y_bin_true, y_bin_pred)
    pob = bin_acc + bin_f1

    dt_acc = exact_match_accuracy_multilabel(y_dt_true, y_dt_pred)
    dt_f1 = micro_f1_multilabel(y_dt_true, y_dt_pred)
    pom = dt_acc + dt_f1

    return {
        "binary_accuracy": bin_acc,
        "binary_micro_f1": bin_f1,
        "PoB_proxy": pob,
        "dt_exact_match_accuracy": dt_acc,
        "dt_micro_f1": dt_f1,
        "PoM_proxy": pom,
        "Score_proxy": 0.6 * pob + 0.4 * pom,
    }


# ============================================================
# Training
# ============================================================
@dataclass
class TrainConfig:
    run_name: str
    teacher_run_name: str | None = None
    teacher_checkpoint_path: str | None = None
    pseudo_csv: str = ""
    device: str = "auto"
    seed: int = 42
    num_workers: int = 8

    train_labels_csv: str = str(TRAIN_LABELS_CSV)
    roi_train_dir: str = str(ROI_TRAIN_DIR)
    roi_unlabeled_dir: str = str(ROI_UNLABELED_SAMPLE_DIR)

    epochs: int = 20
    batch_size: int = 16
    lr: float = 3e-4
    weight_decay: float = 1e-4
    dropout: float = 0.2
    amp: bool = True
    pin_memory: bool = True
    save_every: int = 5

    pseudo_loss_weight: float = 0.3
    dt_pseudo_loss_weight: float = 0.2


def choose_teacher_checkpoint(run_name: str | None, checkpoint_path: str | None) -> Path:
    if checkpoint_path is not None:
        p = Path(checkpoint_path)
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
        return p

    if run_name is None:
        raise ValueError("Provide teacher run_name or teacher checkpoint path.")

    candidates = [
        LOG_DIR / run_name / "supervised_best_train_proxy.pt",
        WEIGHTS_DIR / f"{run_name}_supervised_best_train_proxy.pt",
        LOG_DIR / run_name / "supervised_last.pt",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"Could not find teacher checkpoint for {run_name}")


def load_teacher_backbone(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    dropout = 0.2
    if "config" in ckpt and isinstance(ckpt["config"], dict) and "dropout" in ckpt["config"]:
        dropout = float(ckpt["config"]["dropout"])

    backbone = ResNet18Backbone4Ch()
    model = MultiTaskClassifier(backbone=backbone, feat_dim=backbone.out_dim, dropout=dropout)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.backbone.state_dict(), dropout


def evaluate_on_real_train(model, loader, device):
    model.eval()
    all_y_bin_true = []
    all_y_bin_pred = []
    all_y_dt_true = []
    all_y_dt_pred = []

    with torch.no_grad():
        for x, y_bin, y_dt, _ in loader:
            x = x.to(device, non_blocking=True)
            y_bin = y_bin.to(device, non_blocking=True)
            y_dt = y_dt.to(device, non_blocking=True)

            bin_logit, dt_logit = model(x)
            bin_pred = (torch.sigmoid(bin_logit) >= 0.5).long().cpu().numpy().reshape(-1)
            dt_pred = (torch.sigmoid(dt_logit) >= 0.5).long().cpu().numpy()

            all_y_bin_true.append(y_bin.cpu().numpy().reshape(-1).astype(np.int64))
            all_y_bin_pred.append(bin_pred)
            all_y_dt_true.append(y_dt.cpu().numpy().astype(np.int64))
            all_y_dt_pred.append(dt_pred.astype(np.int64))

    y_bin_true = np.concatenate(all_y_bin_true, axis=0)
    y_bin_pred = np.concatenate(all_y_bin_pred, axis=0)
    y_dt_true = np.concatenate(all_y_dt_true, axis=0)
    y_dt_pred = np.concatenate(all_y_dt_pred, axis=0)

    return challenge_proxy_score(y_bin_true, y_bin_pred, y_dt_true, y_dt_pred)


def train_with_pseudo(
    model,
    train_loader,
    eval_loader,
    optimizer,
    scaler,
    device,
    logger,
    metrics_jsonl,
    cfg,
    run_dir,
):
    use_amp = cfg.amp and device.type == "cuda"

    # pos weights from real data only
    real_rows = []
    train_df = pd.read_csv(cfg.train_labels_csv)
    real_df = load_train_labels(Path(cfg.train_labels_csv))
    defect_pos = real_df["defect"].mean()
    dt_pos = real_df[["DT1", "DT2", "DT3"]].mean().values

    eps = 1e-6
    bin_pos_weight = torch.tensor([(1.0 - defect_pos + eps) / (defect_pos + eps)], dtype=torch.float32, device=device)
    dt_pos_weight = torch.tensor((1.0 - dt_pos + eps) / (dt_pos + eps), dtype=torch.float32, device=device)

    bin_criterion = nn.BCEWithLogitsLoss(pos_weight=bin_pos_weight, reduction="none")
    dt_criterion = nn.BCEWithLogitsLoss(pos_weight=dt_pos_weight, reduction="none")

    logger.info(f"[PSEUDO] binary_pos_weight={bin_pos_weight.detach().cpu().numpy().tolist()}")
    logger.info(f"[PSEUDO] dt_pos_weight={dt_pos_weight.detach().cpu().numpy().tolist()}")

    best_proxy = -1.0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0 = time.time()

        total_loss_sum = 0.0
        n_samples = 0

        pbar = tqdm(train_loader, desc=f"PSEUDO Epoch {epoch}/{cfg.epochs}", leave=False)
        for x, y_bin, y_dt, is_pseudo, stems in pbar:
            x = x.to(device, non_blocking=True)
            y_bin = y_bin.to(device, non_blocking=True)
            y_dt = y_dt.to(device, non_blocking=True)
            is_pseudo = is_pseudo.to(device, non_blocking=True)  # shape (B,1)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                bin_logit, dt_logit = model(x)

                # binary loss per sample
                bin_loss_per = bin_criterion(bin_logit, y_bin).mean(dim=1, keepdim=True)

                # DT loss per sample, handle -1 labels for pseudo binary-only rows
                dt_valid_mask = (y_dt >= 0).float()
                y_dt_clamped = torch.clamp(y_dt, min=0.0, max=1.0)
                dt_loss_raw = dt_criterion(dt_logit, y_dt_clamped) * dt_valid_mask

                dt_valid_count = dt_valid_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
                dt_loss_per = dt_loss_raw.sum(dim=1, keepdim=True) / dt_valid_count

                # weights
                sample_weight_bin = torch.where(
                    is_pseudo > 0.5,
                    torch.full_like(is_pseudo, cfg.pseudo_loss_weight),
                    torch.ones_like(is_pseudo),
                )

                sample_weight_dt = torch.where(
                    is_pseudo > 0.5,
                    torch.full_like(is_pseudo, cfg.dt_pseudo_loss_weight),
                    torch.ones_like(is_pseudo),
                )

                loss = (sample_weight_bin * bin_loss_per).mean() + (sample_weight_dt * dt_loss_per).mean()

            if scaler is not None and use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            bsz = x.size(0)
            total_loss_sum += loss.item() * bsz
            n_samples += bsz
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss_sum / max(1, n_samples)
        train_metrics = evaluate_on_real_train(model, eval_loader, device)
        elapsed = time.time() - t0

        payload = {
            "stage": "pseudo_train",
            "epoch": epoch,
            "loss_total": avg_loss,
            "seconds": elapsed,
            **train_metrics,
        }
        metrics_jsonl.write(payload)

        logger.info(
            "[PSEUDO] epoch=%03d loss=%.6f train_bin_acc=%.4f train_bin_f1=%.4f "
            "train_dt_exact=%.4f train_dt_f1=%.4f train_proxy=%.4f time=%.1fs"
            % (
                epoch,
                avg_loss,
                train_metrics["binary_accuracy"],
                train_metrics["binary_micro_f1"],
                train_metrics["dt_exact_match_accuracy"],
                train_metrics["dt_micro_f1"],
                train_metrics["Score_proxy"],
                elapsed,
            )
        )

        ckpt = {
            "epoch": epoch,
            "stage": "pseudo_train",
            "config": asdict(cfg),
            "model_state_dict": model.state_dict(),
            "backbone_state_dict": model.backbone.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_metrics": train_metrics,
        }

        torch.save(ckpt, run_dir / "pseudo_last.pt")

        if train_metrics["Score_proxy"] > best_proxy:
            best_proxy = train_metrics["Score_proxy"]
            torch.save(ckpt, run_dir / "pseudo_best_train_proxy.pt")
            logger.info(f"[PSEUDO] new best train proxy: {best_proxy:.6f}")

        if epoch % cfg.save_every == 0:
            torch.save(ckpt, run_dir / f"pseudo_epoch_{epoch:03d}.pt")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--run_name", type=str, default=f"pseudo_{now_str()}")
    parser.add_argument("--teacher_run_name", type=str, default=None)
    parser.add_argument("--teacher_checkpoint_path", type=str, default=None)
    parser.add_argument("--pseudo_csv", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--train_labels_csv", type=str, default=str(TRAIN_LABELS_CSV))
    parser.add_argument("--roi_train_dir", type=str, default=str(ROI_TRAIN_DIR))
    parser.add_argument("--roi_unlabeled_dir", type=str, default=str(ROI_UNLABELED_SAMPLE_DIR))

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--pseudo_loss_weight", type=float, default=0.3)
    parser.add_argument("--dt_pseudo_loss_weight", type=float, default=0.2)

    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--save_every", type=int, default=5)

    args = parser.parse_args()

    amp = True
    if args.no_amp:
        amp = False
    elif args.amp:
        amp = True

    return TrainConfig(
        run_name=args.run_name,
        teacher_run_name=args.teacher_run_name,
        teacher_checkpoint_path=args.teacher_checkpoint_path,
        pseudo_csv=args.pseudo_csv,
        device=args.device,
        seed=args.seed,
        num_workers=args.num_workers,
        train_labels_csv=args.train_labels_csv,
        roi_train_dir=args.roi_train_dir,
        roi_unlabeled_dir=args.roi_unlabeled_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        amp=amp,
        pseudo_loss_weight=args.pseudo_loss_weight,
        dt_pseudo_loss_weight=args.dt_pseudo_loss_weight,
        save_every=args.save_every,
    )


def main():
    cfg = parse_args()
    seed_everything(cfg.seed)

    run_dir = LOG_DIR / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_run_logger(run_dir)
    metrics_jsonl = JsonlLogger(run_dir / "metrics.jsonl")
    failure_logger = JsonlLogger(run_dir / "failures.jsonl")

    try:
        device = resolve_device(cfg.device)
        logger.info(f"Using device: {device}")
        logger.info(f"Config:\n{json.dumps(asdict(cfg), indent=2)}")

        real_df = load_train_labels(Path(cfg.train_labels_csv))
        pseudo_df = load_pseudo_labels(Path(cfg.pseudo_csv))

        logger.info(f"Real labeled rows: {len(real_df)}")
        logger.info(f"Pseudo-labeled rows: {len(pseudo_df)}")

        train_ds = MixedROITensorDataset(
            real_labels_df=real_df,
            pseudo_labels_df=pseudo_df,
            roi_train_dir=Path(cfg.roi_train_dir),
            roi_unlabeled_dir=Path(cfg.roi_unlabeled_dir),
            transform=ROITrainAugment(),
            failure_logger=failure_logger,
        )

        eval_ds = RealOnlyEvalDataset(
            labels_df=real_df,
            roi_train_dir=Path(cfg.roi_train_dir),
        )

        num_workers = get_num_workers(cfg.num_workers)
        common = dict(
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(num_workers > 0),
        )

        n_real = sum(1 for r in train_ds.records if r["is_pseudo"] == 0)
        n_pseudo = sum(1 for r in train_ds.records if r["is_pseudo"] == 1)
        real_weight = n_pseudo / max(n_real, 1)  # ~246x for your dataset
        sample_weights = [
            real_weight if r["is_pseudo"] == 0 else 1.0
            for r in train_ds.records
        ]
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_ds),
            replacement=True,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            sampler=sampler,        # replaces shuffle=True
            drop_last=False,
            **common,
        )

        eval_loader = DataLoader(
            eval_ds,
            batch_size=max(cfg.batch_size, 32),
            shuffle=False,
            drop_last=False,
            **common,
        )

        teacher_ckpt = choose_teacher_checkpoint(cfg.teacher_run_name, cfg.teacher_checkpoint_path)
        teacher_backbone_state, teacher_dropout = load_teacher_backbone(teacher_ckpt, device)

        backbone = ResNet18Backbone4Ch()
        backbone.load_state_dict(teacher_backbone_state)

        model = MultiTaskClassifier(
            backbone=backbone,
            feat_dim=backbone.out_dim,
            dropout=cfg.dropout if cfg.dropout is not None else teacher_dropout,
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device.type == "cuda"))

        train_with_pseudo(
            model=model,
            train_loader=train_loader,
            eval_loader=eval_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            logger=logger,
            metrics_jsonl=metrics_jsonl,
            cfg=cfg,
            run_dir=run_dir,
        )

        best_src = run_dir / "pseudo_best_train_proxy.pt"
        final_path = WEIGHTS_DIR / f"{cfg.run_name}_pseudo_best_train_proxy.pt"
        if best_src.exists():
            final_path.write_bytes(best_src.read_bytes())
            logger.info(f"Copied best pseudo-trained weights to {final_path}")

        logger.info("Pseudo-label training completed successfully.")

    except Exception as e:
        failure_logger.write({
            "stage": "main",
            "error": str(e),
            "traceback": traceback.format_exc(),
        })
        logger.exception("Fatal error during pseudo-label training.")
        raise


if __name__ == "__main__":
    main()