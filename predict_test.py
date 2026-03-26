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
OUTPUT_CSV = DATA_ROOT / "submission_test.csv"

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
            4, old_conv.out_channels,
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
    def __init__(self, backbone, feat_dim=512, dropout=0.2):
        super().__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(dropout)
        self.bin_head = nn.Linear(feat_dim, 1)
        self.dt_head = nn.Linear(feat_dim, 3)

    def forward(self, x):
        feat = self.dropout(self.backbone(x))
        return self.bin_head(feat), self.dt_head(feat)


# ============================================================
# Dataset
# ============================================================
class TestDataset(Dataset):
    def __init__(self, roi_dir: Path):
        self.files = sorted(roi_dir.glob("*.npy"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        arr = np.load(path)

        x = torch.from_numpy(arr).float() / 255.0
        image_id = path.stem + ".jpg"

        return x, image_id


# ============================================================
# Load checkpoint
# ============================================================
def load_model(run_name, device):
    candidates = [
        LOG_DIR / run_name / "supervised_best_train_proxy.pt",
        WEIGHTS_DIR / f"{run_name}_supervised_best_train_proxy.pt",
        LOG_DIR / run_name / "supervised_last.pt",
    ]

    for c in candidates:
        if c.exists():
            ckpt = torch.load(c, map_location=device)
            break
    else:
        raise FileNotFoundError(f"No checkpoint found for {run_name}")

    backbone = ResNet18Backbone4Ch()
    model = MultiTaskClassifier(backbone)

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    return model


# ============================================================
# Prediction
# ============================================================
@torch.no_grad()
def run_inference(model, loader, device, thresholds):
    results = {}

    for x, image_ids in tqdm(loader):
        x = x.to(device)

        bin_logit, dt_logit = model(x)

        p_def = torch.sigmoid(bin_logit).squeeze(1).cpu().numpy()
        p_dt = torch.sigmoid(dt_logit).cpu().numpy()

        for i, img_id in enumerate(image_ids):
            d = int(p_def[i] >= thresholds[0])
            dt1 = int(p_dt[i, 0] >= thresholds[1])
            dt2 = int(p_dt[i, 1] >= thresholds[2])
            dt3 = int(p_dt[i, 2] >= thresholds[3])

            if d == 0:
                dt1, dt2, dt3 = 0, 0, 0

            results[img_id] = (d, dt1, dt2, dt3)

    return results


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default="auto")

    parser.add_argument("--defect_threshold", type=float, default=0.5)
    parser.add_argument("--dt1_threshold", type=float, default=0.5)
    parser.add_argument("--dt2_threshold", type=float, default=0.5)
    parser.add_argument("--dt3_threshold", type=float, default=0.5)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(args.run_name, device)

    dataset = TestDataset(ROI_TEST_DIR)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    thresholds = (
        args.defect_threshold,
        args.dt1_threshold,
        args.dt2_threshold,
        args.dt3_threshold,
    )

    preds = run_inference(model, loader, device, thresholds)

    # ============================================================
    # Fill template
    # ============================================================
    df = pd.read_csv(TEMPLATE_CSV)

    for i in range(len(df)):
        img_id = df.loc[i, "Image_id"]

        if img_id not in preds:
            continue

        d, dt1, dt2, dt3 = preds[img_id]

        df.loc[i, "Defect"] = d
        df.loc[i, "DT1(Missing_Perforations)"] = dt1
        df.loc[i, "DT2(Touching_Perforations)"] = dt2
        df.loc[i, "DT3(Out_Of_Bounds)"] = dt3

    df.to_csv(OUTPUT_CSV, index=False)

    print(f"Saved submission to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()