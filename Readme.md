# QCRE Data Challenge 2026

Automated defect detection on pet food packaging perforations.  
**Competition score: 1.71492** — IISE Annual Conference, May 2026.

The pipeline detects two things from grayscale inspection images:

- **Binary pass/fail** — is there a defect?
- **Defect type** — which of DT1 (misaligned), DT2 (torn), or DT3 (out-of-bounds) perforation

Scoring: `Score = 0.6 × PoB + 0.4 × PoM` (max = 4.0)

---

## Repository structure

```
QCRE_Datachalleng2026/
├── DataFile/
│   ├── test_data/               ← place your test images here
│   ├── train_data/
│   ├── validation_data/
│   ├── unlabeled_extracted/
│   ├── roi_data/                ← created by config.py
│   ├── trained_weights/         ← model checkpoints
│   └── train_labels.csv
├── config.py                    ← path definitions + dir setup
├── build_roi_dataset.py         ← ROI extractor
├── train.py                     ← SSL pretraining + supervised fine-tuning
├── train_with_pseudo_labels.py  ← pseudo-label training
├── generate_pseudo_labels.py    ← generate pseudo labels from unlabeled data
├── predict_test.py              ← inference on test set → submission CSV
├── predict_validation.py        ← inference on validation set
└── debug_extracted_samples.py   ← visualize extracted ROIs
```

---

## Inference quickstart (test set)

Follow these four steps in order.

### Step 1 — Place test images

Copy all test `.jpg` images into:

```
DataFile/test_data/
```

No subfolders — all images flat in that directory.

---

### Step 2 — Create project directories

Run `config.py` once to create all required subfolders under `DataFile/`:

```bash
python config.py
```

This creates `roi_data/`, `roi_data/test/`, `trained_weights/`, `logs/`, and other required directories. Safe to re-run; existing directories are not overwritten.

---

### Step 3 — Extract ROIs from test images

Run the ROI extractor targeting the test split:

```bash
python build_roi_dataset.py \
  --only_split test \
  --workers 8
```

This processes every image in `DataFile/test_data/`, detects the four perforation panel regions per image, and saves a 4-channel `.npy` tensor for each image into `DataFile/roi_data/test/`. If detection fails for an image, a learned fallback template is used automatically.

**Optional flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--workers N` | `8` | Parallel worker processes |
| `--save_debug` | off | Save mosaic and bounding-box visualizations to `roi_data/roi_box_debug/test/` |
| `--rebuild_fallback` | off | Recompute the fallback template from scratch |
| `--template_source PATH` | `DataFile/train_data` | Source directory for computing the fallback template |

---

### Step 4 — Run inference

```bash
python predict_test.py \
  --run_name sample_ssl_sup \
  --roi_test_dir DataFile/roi_data/test \
  --template_csv DataFile/test_detection.csv \
  --output_csv DataFile/submission_test.csv \
  --device auto \
  --defect_threshold 0.42 \
  --dt1_threshold 0.38 \
  --dt2_threshold 0.61 \
  --dt3_threshold 0.47
```

The submission CSV is written to `DataFile/submission_test.csv`.

**Argument reference:**

| Argument | Description |
|----------|-------------|
| `--run_name` | Name of the trained model run to load from `DataFile/trained_weights/` |
| `--roi_test_dir` | Path to the extracted test ROI tensors (output of Step 3) |
| `--template_csv` | Competition-provided test image list CSV |
| `--output_csv` | Path where the submission CSV will be written |
| `--device` | `auto` (uses CUDA if available), `cuda`, or `cpu` |
| `--defect_threshold` | Binary pass/fail decision threshold (default tuned to 0.42) |
| `--dt1_threshold` | DT1 (misaligned perforation) decision threshold |
| `--dt2_threshold` | DT2 (torn perforation) decision threshold |
| `--dt3_threshold` | DT3 (out-of-bounds perforation) decision threshold |

> **Note on thresholds:** The four thresholds above were tuned on the validation set and differ from the default 0.5. Do not change them unless you have a validation set to evaluate against.

---

## Model overview

The model is a two-stage pipeline:

1. **SSL pretraining** — A ResNet-18 backbone (modified for 4-channel input) is pretrained on ~48,000 unlabeled images using SimCLR contrastive learning (NT-Xent loss) to learn visual representations without labels.

2. **Supervised fine-tuning** — The pretrained backbone is connected to two independent linear heads (binary pass/fail + three-class DT multi-label) and fine-tuned on 195 labeled training samples with class-weighted BCE losses.

Each image is represented as a 4-channel ROI tensor (224×224 per channel), where each channel corresponds to one of the four perforation panels extracted from the combined inspection frame.

---

## Dependencies

```bash
pip install torch torchvision opencv-python pillow numpy pandas tqdm
```

A CUDA-capable GPU is recommended. The pipeline was developed and tested on an HPC cluster running PyTorch 2.x with CUDA.

---

## Other scripts

| Script | Purpose |
|--------|---------|
| `train.py` | Full training pipeline: SSL pretraining → supervised fine-tuning |
| `train_with_pseudo_labels.py` | Fine-tune with pseudo-labeled unlabeled data added to training |
| `generate_pseudo_labels.py` | Run a trained model over unlabeled data to generate pseudo labels |
| `predict_validation.py` | Inference on the validation set with proxy score evaluation |
| `debug_extracted_samples.py` | Visualize ROI extraction results for quality checking |
