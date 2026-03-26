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

from config import (
    LOG_DIR,
    WEIGHTS_DIR,
    TRAIN_LABELS_CSV,
    ROI_TRAIN_DIR,
    ensure_project_dirs,
    SEED,
)

# ------------------------------------------------------------
# NEW: sample unlabeled ROI directory
# ------------------------------------------------------------
ROI_UNLABELED_SAMPLE_DIR = Path(__file__).resolve().parent / "DataFile" / "roi_data" / "unlabeled_sample"


# =============================================================================
# Small utilities
# =============================================================================

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def now_str() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_arg)


def get_num_workers(requested: int) -> int:
    if requested >= 0:
        return requested
    cpu_count = os.cpu_count() or 4
    return max(2, min(8, cpu_count // 2))


# =============================================================================
# Logging
# =============================================================================

class JsonlLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, payload: Dict):
        with self.path.open("a") as f:
            f.write(json.dumps(payload) + "\n")


def setup_run_logger(run_dir: Path) -> logging.Logger:
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"qcre_train_{run_dir.name}")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(run_dir / "train.log")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


# =============================================================================
# Label parsing
# =============================================================================

def _pick_existing_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def load_train_labels(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    image_col = _pick_existing_col(df, [
        "image_id", "Image_id", "filename", "file_name", "img", "image"
    ])
    defect_col = _pick_existing_col(df, [
        "defect", "Defect", "label", "pass_fail", "target"
    ])
    dt1_col = _pick_existing_col(df, [
        "DT1", "dt1", "DT1_MP", "dt1_mp"
    ])
    dt2_col = _pick_existing_col(df, [
        "DT2", "dt2", "DT2_TP", "dt2_tp"
    ])
    dt3_col = _pick_existing_col(df, [
        "DT3", "dt3", "DT3_OOB", "dt3_oob"
    ])

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
            f"Could not infer required label columns from {csv_path}. "
            f"Missing: {missing}. Found columns: {list(df.columns)}"
        )

    out = pd.DataFrame({
        "image_id": df[image_col].astype(str),
        "defect": df[defect_col].astype(int),
        "DT1": df[dt1_col].astype(int),
        "DT2": df[dt2_col].astype(int),
        "DT3": df[dt3_col].astype(int),
    })

    out["stem"] = out["image_id"].map(lambda x: Path(x).stem)
    return out


# =============================================================================
# ROI transforms
# =============================================================================

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

        if self.brightness > 0:
            b = 1.0 + (torch.rand(1).item() * 2.0 - 1.0) * self.brightness
            x = x * b

        if self.contrast > 0:
            mean = x.mean(dim=(1, 2), keepdim=True)
            c = 1.0 + (torch.rand(1).item() * 2.0 - 1.0) * self.contrast
            x = (x - mean) * c + mean

        if self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std

        x = torch.clamp(x, 0.0, 1.0)
        return x


class ROIIdentity:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x


# =============================================================================
# Datasets
# =============================================================================

class SSLROITensorDataset(Dataset):
    def __init__(
        self,
        directories: List[Path],
        aug1=None,
        aug2=None,
        failure_logger: Optional[JsonlLogger] = None,
    ):
        self.files: List[Path] = []
        for d in directories:
            if d.exists():
                self.files.extend(sorted(d.glob("*.npy")))
        self.files = sorted(self.files)
        self.aug1 = aug1 if aug1 is not None else ROITrainAugment()
        self.aug2 = aug2 if aug2 is not None else ROITrainAugment()
        self.failure_logger = failure_logger

        if len(self.files) == 0:
            raise RuntimeError(f"No .npy ROI files found in: {directories}")

    def __len__(self):
        return len(self.files)

    def _load_tensor(self, path: Path) -> torch.Tensor:
        try:
            arr = np.load(path)
            if arr.ndim != 3 or arr.shape[0] != 4:
                raise ValueError(f"Unexpected ROI tensor shape {arr.shape}")
            x = torch.from_numpy(arr).float() / 255.0
            return x
        except Exception as e:
            if self.failure_logger is not None:
                self.failure_logger.write({
                    "stage": "ssl_dataset",
                    "file": str(path),
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                })
            raise

    def __getitem__(self, idx: int):
        path = self.files[idx]
        x = self._load_tensor(path)
        v1 = self.aug1(x.clone())
        v2 = self.aug2(x.clone())
        return v1, v2, path.stem


class SupervisedROITensorDataset(Dataset):
    def __init__(
        self,
        roi_dir: Path,
        labels_df: pd.DataFrame,
        transform=None,
        failure_logger: Optional[JsonlLogger] = None,
    ):
        self.roi_dir = roi_dir
        self.transform = transform if transform is not None else ROIIdentity()
        self.failure_logger = failure_logger

        records = []
        missing = []
        for _, row in labels_df.iterrows():
            stem = row["stem"]
            npy_path = roi_dir / f"{stem}.npy"
            if npy_path.exists():
                records.append({
                    "stem": stem,
                    "path": npy_path,
                    "defect": int(row["defect"]),
                    "dt": np.array([int(row["DT1"]), int(row["DT2"]), int(row["DT3"])], dtype=np.float32),
                })
            else:
                missing.append(stem)

        self.records = records
        self.missing = missing

        if len(self.records) == 0:
            raise RuntimeError(f"No labeled ROI tensors found in {roi_dir}")

    def __len__(self):
        return len(self.records)

    def _load_tensor(self, path: Path) -> torch.Tensor:
        try:
            arr = np.load(path)
            if arr.ndim != 3 or arr.shape[0] != 4:
                raise ValueError(f"Unexpected ROI tensor shape {arr.shape}")
            x = torch.from_numpy(arr).float() / 255.0
            return x
        except Exception as e:
            if self.failure_logger is not None:
                self.failure_logger.write({
                    "stage": "supervised_dataset",
                    "file": str(path),
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                })
            raise

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        x = self._load_tensor(rec["path"])
        x = self.transform(x)
        y_bin = torch.tensor([rec["defect"]], dtype=torch.float32)
        y_dt = torch.tensor(rec["dt"], dtype=torch.float32)
        return x, y_bin, y_dt, rec["stem"]


# =============================================================================
# Models
# =============================================================================

class ResNet18Backbone4Ch(nn.Module):
    def __init__(self, pretrained: bool = False):
        super().__init__()
        model = resnet18(weights=None if not pretrained else None)
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


class SimSiamSSL(nn.Module):
    def __init__(self, backbone: nn.Module, feat_dim: int = 512, proj_dim: int = 2048, pred_dim: int = 512):
        super().__init__()
        self.backbone = backbone
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, proj_dim, bias=False),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim, bias=False),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim, bias=False),
            nn.BatchNorm1d(proj_dim, affine=False),
        )
        self.predictor = nn.Sequential(
            nn.Linear(proj_dim, pred_dim, bias=False),
            nn.BatchNorm1d(pred_dim),
            nn.ReLU(inplace=True),
            nn.Linear(pred_dim, proj_dim),
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        f1 = self.backbone(x1)
        f2 = self.backbone(x2)

        z1 = self.projector(f1)
        z2 = self.projector(f2)

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)
        return p1, p2, z1.detach(), z2.detach()


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


# =============================================================================
# Metrics
# =============================================================================

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


def challenge_proxy_score(
    y_bin_true: np.ndarray,
    y_bin_pred: np.ndarray,
    y_dt_true: np.ndarray,
    y_dt_pred: np.ndarray,
) -> Dict[str, float]:
    bin_acc = float((y_bin_true == y_bin_pred).mean())
    bin_f1 = micro_f1_binary(y_bin_true, y_bin_pred)
    pob = bin_acc + bin_f1

    dt_acc = exact_match_accuracy_multilabel(y_dt_true, y_dt_pred)
    dt_f1 = micro_f1_multilabel(y_dt_true, y_dt_pred)
    pom = dt_acc + dt_f1

    score = 0.6 * pob + 0.4 * pom

    return {
        "binary_accuracy": bin_acc,
        "binary_micro_f1": bin_f1,
        "PoB_proxy": pob,
        "dt_exact_match_accuracy": dt_acc,
        "dt_micro_f1": dt_f1,
        "PoM_proxy": pom,
        "Score_proxy": score,
    }


# =============================================================================
# Losses
# =============================================================================

def negative_cosine_similarity(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    p = F.normalize(p, dim=1)
    z = F.normalize(z, dim=1)
    return -(p * z).sum(dim=1).mean()


# =============================================================================
# Training loops
# =============================================================================

@dataclass
class TrainConfig:
    run_name: str
    device: str = "auto"
    seed: int = SEED
    num_workers: int = -1

    train_labels_csv: str = str(TRAIN_LABELS_CSV)
    roi_train_dir: str = str(ROI_TRAIN_DIR)
    roi_unlabeled_dir: str = str(ROI_UNLABELED_SAMPLE_DIR)

    ssl_epochs: int = 50
    ssl_batch_size: int = 128
    ssl_lr: float = 1e-3
    ssl_weight_decay: float = 1e-4

    sup_epochs: int = 40
    sup_batch_size: int = 16
    sup_lr: float = 3e-4
    sup_weight_decay: float = 1e-4
    binary_loss_weight: float = 1.0
    dt_loss_weight: float = 1.0
    dropout: float = 0.2

    amp: bool = True
    pin_memory: bool = True
    save_every: int = 5


def train_ssl(
    model: SimSiamSSL,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    device: torch.device,
    logger: logging.Logger,
    metrics_jsonl: JsonlLogger,
    cfg: TrainConfig,
    run_dir: Path,
):
    model.train()
    use_amp = cfg.amp and device.type == "cuda"

    for epoch in range(1, cfg.ssl_epochs + 1):
        epoch_loss = 0.0
        n_samples = 0
        t0 = time.time()

        pbar = tqdm(loader, desc=f"SSL Epoch {epoch}/{cfg.ssl_epochs}", leave=False)
        for x1, x2, stems in pbar:
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)
            bsz = x1.size(0)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                p1, p2, z1, z2 = model(x1, x2)
                loss = 0.5 * (
                    negative_cosine_similarity(p1, z2) +
                    negative_cosine_similarity(p2, z1)
                )

            if scaler is not None and use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item() * bsz
            n_samples += bsz
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / max(1, n_samples)
        elapsed = time.time() - t0

        payload = {
            "stage": "ssl",
            "epoch": epoch,
            "loss": avg_loss,
            "seconds": elapsed,
            "samples": n_samples,
        }
        metrics_jsonl.write(payload)
        logger.info(f"[SSL] epoch={epoch:03d} loss={avg_loss:.6f} time={elapsed:.1f}s")

        if (epoch % cfg.save_every == 0) or (epoch == cfg.ssl_epochs):
            ckpt_path = run_dir / f"ssl_epoch_{epoch:03d}.pt"
            torch.save({
                "epoch": epoch,
                "stage": "ssl",
                "config": asdict(cfg),
                "backbone_state_dict": model.backbone.state_dict(),
                "ssl_model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, ckpt_path)


def evaluate_on_train(
    model: MultiTaskClassifier,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()

    all_y_bin_true = []
    all_y_bin_pred = []
    all_y_dt_true = []
    all_y_dt_pred = []

    with torch.no_grad():
        for x, y_bin, y_dt, stems in loader:
            x = x.to(device, non_blocking=True)
            y_bin = y_bin.to(device, non_blocking=True)
            y_dt = y_dt.to(device, non_blocking=True)

            bin_logit, dt_logit = model(x)
            bin_pred = (torch.sigmoid(bin_logit) >= 0.5).long().cpu().numpy().reshape(-1)
            dt_pred = (torch.sigmoid(dt_logit) >= 0.5).long().cpu().numpy()

            all_y_bin_true.append(y_bin.cpu().numpy().reshape(-1).astype(np.int64))
            all_y_bin_pred.append(bin_pred.astype(np.int64))
            all_y_dt_true.append(y_dt.cpu().numpy().astype(np.int64))
            all_y_dt_pred.append(dt_pred.astype(np.int64))

    y_bin_true = np.concatenate(all_y_bin_true, axis=0)
    y_bin_pred = np.concatenate(all_y_bin_pred, axis=0)
    y_dt_true = np.concatenate(all_y_dt_true, axis=0)
    y_dt_pred = np.concatenate(all_y_dt_pred, axis=0)

    return challenge_proxy_score(y_bin_true, y_bin_pred, y_dt_true, y_dt_pred)


def train_supervised(
    model: MultiTaskClassifier,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    device: torch.device,
    logger: logging.Logger,
    metrics_jsonl: JsonlLogger,
    cfg: TrainConfig,
    run_dir: Path,
):
    use_amp = cfg.amp and device.type == "cuda"
    best_proxy = -1.0

    dt_sum = np.zeros(3, dtype=np.float64)
    n_rows = 0
    defect_sum = 0.0
    for _, y_bin, y_dt, _ in train_loader:
        defect_sum += y_bin.sum().item()
        dt_sum += y_dt.sum(dim=0).numpy()
        n_rows += y_bin.size(0)

    eps = 1e-6
    defect_pos = defect_sum / max(1, n_rows)
    dt_pos = dt_sum / max(1, n_rows)

    bin_pos_weight = torch.tensor([(1.0 - defect_pos + eps) / (defect_pos + eps)], dtype=torch.float32, device=device)
    dt_pos_weight = torch.tensor((1.0 - dt_pos + eps) / (dt_pos + eps), dtype=torch.float32, device=device)

    bin_criterion = nn.BCEWithLogitsLoss(pos_weight=bin_pos_weight)
    dt_criterion = nn.BCEWithLogitsLoss(pos_weight=dt_pos_weight)

    logger.info(f"[SUP] binary_pos_weight={bin_pos_weight.detach().cpu().numpy().tolist()}")
    logger.info(f"[SUP] dt_pos_weight={dt_pos_weight.detach().cpu().numpy().tolist()}")

    for epoch in range(1, cfg.sup_epochs + 1):
        model.train()
        t0 = time.time()
        running_total = 0.0
        running_bin = 0.0
        running_dt = 0.0
        n_samples = 0

        pbar = tqdm(train_loader, desc=f"SUP Epoch {epoch}/{cfg.sup_epochs}", leave=False)
        for x, y_bin, y_dt, stems in pbar:
            x = x.to(device, non_blocking=True)
            y_bin = y_bin.to(device, non_blocking=True)
            y_dt = y_dt.to(device, non_blocking=True)
            bsz = x.size(0)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                bin_logit, dt_logit = model(x)
                loss_bin = bin_criterion(bin_logit, y_bin)
                loss_dt = dt_criterion(dt_logit, y_dt)
                loss = cfg.binary_loss_weight * loss_bin + cfg.dt_loss_weight * loss_dt

            if scaler is not None and use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            running_total += loss.item() * bsz
            running_bin += loss_bin.item() * bsz
            running_dt += loss_dt.item() * bsz
            n_samples += bsz

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                l_bin=f"{loss_bin.item():.4f}",
                l_dt=f"{loss_dt.item():.4f}",
            )

        avg_total = running_total / max(1, n_samples)
        avg_bin = running_bin / max(1, n_samples)
        avg_dt = running_dt / max(1, n_samples)

        train_metrics = evaluate_on_train(model, eval_loader, device)
        elapsed = time.time() - t0

        payload = {
            "stage": "supervised",
            "epoch": epoch,
            "loss_total": avg_total,
            "loss_binary": avg_bin,
            "loss_dt": avg_dt,
            "seconds": elapsed,
            **train_metrics,
        }
        metrics_jsonl.write(payload)

        logger.info(
            "[SUP] epoch=%03d loss=%.6f bin_loss=%.6f dt_loss=%.6f "
            "train_bin_acc=%.4f train_bin_f1=%.4f train_dt_exact=%.4f train_dt_f1=%.4f train_proxy=%.4f time=%.1fs"
            % (
                epoch,
                avg_total,
                avg_bin,
                avg_dt,
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
            "stage": "supervised",
            "config": asdict(cfg),
            "model_state_dict": model.state_dict(),
            "backbone_state_dict": model.backbone.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_metrics": train_metrics,
        }

        last_path = run_dir / "supervised_last.pt"
        torch.save(ckpt, last_path)

        if train_metrics["Score_proxy"] > best_proxy:
            best_proxy = train_metrics["Score_proxy"]
            best_path = run_dir / "supervised_best_train_proxy.pt"
            torch.save(ckpt, best_path)
            logger.info(f"[SUP] new best train proxy score: {best_proxy:.6f}")

        if epoch % cfg.save_every == 0:
            torch.save(ckpt, run_dir / f"supervised_epoch_{epoch:03d}.pt")


# =============================================================================
# Main
# =============================================================================

def build_loaders(
    cfg: TrainConfig,
    failure_logger: JsonlLogger,
    logger: logging.Logger,
):
    labels_df = load_train_labels(Path(cfg.train_labels_csv))

    ssl_dirs = [Path(cfg.roi_unlabeled_dir), Path(cfg.roi_train_dir)]
    for d in ssl_dirs:
        logger.info(f"SSL source dir: {d} | exists={d.exists()}")

    ssl_ds = SSLROITensorDataset(
        directories=ssl_dirs,
        aug1=ROITrainAugment(),
        aug2=ROITrainAugment(),
        failure_logger=failure_logger,
    )

    sup_train_ds = SupervisedROITensorDataset(
        roi_dir=Path(cfg.roi_train_dir),
        labels_df=labels_df,
        transform=ROITrainAugment(),
        failure_logger=failure_logger,
    )

    sup_eval_ds = SupervisedROITensorDataset(
        roi_dir=Path(cfg.roi_train_dir),
        labels_df=labels_df,
        transform=ROIIdentity(),
        failure_logger=failure_logger,
    )

    logger.info(f"SSL dataset size: {len(ssl_ds)}")
    logger.info(f"Supervised train dataset size: {len(sup_train_ds)}")

    if len(sup_train_ds.missing) > 0:
        logger.warning(f"Missing {len(sup_train_ds.missing)} labeled ROI tensors in {cfg.roi_train_dir}")
        miss_path = Path(LOG_DIR) / cfg.run_name / "missing_labeled_roi_files.txt"
        miss_path.parent.mkdir(parents=True, exist_ok=True)
        miss_path.write_text("\n".join(sup_train_ds.missing))

    num_workers = get_num_workers(cfg.num_workers)
    common = dict(
        num_workers=num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=(num_workers > 0),
    )

    ssl_loader = DataLoader(
        ssl_ds,
        batch_size=cfg.ssl_batch_size,
        shuffle=True,
        drop_last=True,
        **common,
    )

    sup_train_loader = DataLoader(
        sup_train_ds,
        batch_size=cfg.sup_batch_size,
        shuffle=True,
        drop_last=False,
        **common,
    )

    sup_eval_loader = DataLoader(
        sup_eval_ds,
        batch_size=max(cfg.sup_batch_size, 32),
        shuffle=False,
        drop_last=False,
        **common,
    )

    return labels_df, ssl_loader, sup_train_loader, sup_eval_loader


@dataclass
class ArgsConfig:
    run_name: str
    device: str
    seed: int
    num_workers: int
    train_labels_csv: str
    roi_train_dir: str
    roi_unlabeled_dir: str
    ssl_epochs: int
    ssl_batch_size: int
    ssl_lr: float
    ssl_weight_decay: float
    sup_epochs: int
    sup_batch_size: int
    sup_lr: float
    sup_weight_decay: float
    binary_loss_weight: float
    dt_loss_weight: float
    dropout: float
    amp: bool
    pin_memory: bool
    save_every: int


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--run_name", type=str, default=f"run_{now_str()}")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--num_workers", type=int, default=-1)

    parser.add_argument("--train_labels_csv", type=str, default=str(TRAIN_LABELS_CSV))
    parser.add_argument("--roi_train_dir", type=str, default=str(ROI_TRAIN_DIR))
    parser.add_argument("--roi_unlabeled_dir", type=str, default=str(ROI_UNLABELED_SAMPLE_DIR))

    parser.add_argument("--ssl_epochs", type=int, default=50)
    parser.add_argument("--ssl_batch_size", type=int, default=128)
    parser.add_argument("--ssl_lr", type=float, default=1e-3)
    parser.add_argument("--ssl_weight_decay", type=float, default=1e-4)

    parser.add_argument("--sup_epochs", type=int, default=40)
    parser.add_argument("--sup_batch_size", type=int, default=16)
    parser.add_argument("--sup_lr", type=float, default=3e-4)
    parser.add_argument("--sup_weight_decay", type=float, default=1e-4)
    parser.add_argument("--binary_loss_weight", type=float, default=1.0)
    parser.add_argument("--dt_loss_weight", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--no_pin_memory", action="store_true")
    parser.add_argument("--save_every", type=int, default=5)

    args = parser.parse_args()

    amp = True
    if args.no_amp:
        amp = False
    elif args.amp:
        amp = True

    pin_memory = True
    if args.no_pin_memory:
        pin_memory = False
    elif args.pin_memory:
        pin_memory = True

    return TrainConfig(
        run_name=args.run_name,
        device=args.device,
        seed=args.seed,
        num_workers=args.num_workers,
        train_labels_csv=args.train_labels_csv,
        roi_train_dir=args.roi_train_dir,
        roi_unlabeled_dir=args.roi_unlabeled_dir,
        ssl_epochs=args.ssl_epochs,
        ssl_batch_size=args.ssl_batch_size,
        ssl_lr=args.ssl_lr,
        ssl_weight_decay=args.ssl_weight_decay,
        sup_epochs=args.sup_epochs,
        sup_batch_size=args.sup_batch_size,
        sup_lr=args.sup_lr,
        sup_weight_decay=args.sup_weight_decay,
        binary_loss_weight=args.binary_loss_weight,
        dt_loss_weight=args.dt_loss_weight,
        dropout=args.dropout,
        amp=amp,
        pin_memory=pin_memory,
        save_every=args.save_every,
    )


def main():
    cfg = parse_args()
    ensure_project_dirs()
    seed_everything(cfg.seed)

    run_dir = Path(LOG_DIR) / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_run_logger(run_dir)
    metrics_jsonl = JsonlLogger(run_dir / "metrics.jsonl")
    failure_logger = JsonlLogger(run_dir / "failures.jsonl")

    try:
        device = resolve_device(cfg.device)
        logger.info(f"Using device: {device}")
        logger.info(f"Config:\n{json.dumps(asdict(cfg), indent=2)}")

        with (run_dir / "config.json").open("w") as f:
            json.dump(asdict(cfg), f, indent=2)

        _, ssl_loader, sup_train_loader, sup_eval_loader = build_loaders(
            cfg=cfg,
            failure_logger=failure_logger,
            logger=logger,
        )

        backbone = ResNet18Backbone4Ch(pretrained=False)
        ssl_model = SimSiamSSL(backbone=backbone, feat_dim=backbone.out_dim)
        ssl_model.to(device)

        ssl_optimizer = torch.optim.AdamW(
            ssl_model.parameters(),
            lr=cfg.ssl_lr,
            weight_decay=cfg.ssl_weight_decay,
        )
        ssl_scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device.type == "cuda"))

        logger.info("Starting SSL pretraining on roi_data/unlabeled_sample + roi_data/train")
        train_ssl(
            model=ssl_model,
            loader=ssl_loader,
            optimizer=ssl_optimizer,
            scaler=ssl_scaler,
            device=device,
            logger=logger,
            metrics_jsonl=metrics_jsonl,
            cfg=cfg,
            run_dir=run_dir,
        )

        ssl_backbone_path = Path(WEIGHTS_DIR) / f"{cfg.run_name}_ssl_backbone.pt"
        torch.save({
            "config": asdict(cfg),
            "backbone_state_dict": ssl_model.backbone.state_dict(),
        }, ssl_backbone_path)
        logger.info(f"Saved SSL backbone to {ssl_backbone_path}")

        clf_backbone = ResNet18Backbone4Ch(pretrained=False)
        clf_backbone.load_state_dict(ssl_model.backbone.state_dict())

        clf_model = MultiTaskClassifier(
            backbone=clf_backbone,
            feat_dim=clf_backbone.out_dim,
            dropout=cfg.dropout,
        ).to(device)

        sup_optimizer = torch.optim.AdamW(
            clf_model.parameters(),
            lr=cfg.sup_lr,
            weight_decay=cfg.sup_weight_decay,
        )
        sup_scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device.type == "cuda"))

        logger.info("Starting supervised training on roi_data/train only")
        train_supervised(
            model=clf_model,
            train_loader=sup_train_loader,
            eval_loader=sup_eval_loader,
            optimizer=sup_optimizer,
            scaler=sup_scaler,
            device=device,
            logger=logger,
            metrics_jsonl=metrics_jsonl,
            cfg=cfg,
            run_dir=run_dir,
        )

        final_weights_path = Path(WEIGHTS_DIR) / f"{cfg.run_name}_supervised_best_train_proxy.pt"
        best_src = run_dir / "supervised_best_train_proxy.pt"
        if best_src.exists():
            final_weights_path.write_bytes(best_src.read_bytes())
            logger.info(f"Copied best supervised weights to {final_weights_path}")

        logger.info("Training completed successfully.")

    except Exception as e:
        failure_logger.write({
            "stage": "main",
            "error": str(e),
            "traceback": traceback.format_exc(),
        })
        logger.exception("Fatal error during training.")
        raise


if __name__ == "__main__":
    main()