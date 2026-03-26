from pathlib import Path

# ============================================================
# Project root
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "DataFile"

# ============================================================
# Data folders (based on your structure)
# ============================================================
LOG_DIR = DATA_ROOT / "logs"
TRAIN_DIR = DATA_ROOT / "train_data"
VAL_DIR = DATA_ROOT / "validation_data"
TEST_DIR = DATA_ROOT / "test_data"

UNLABELED_ARCHIVES_DIR = DATA_ROOT / "unlabeled_archives"
UNLABELED_DIR = DATA_ROOT / "unlabeled_extracted"

WEIGHTS_DIR = DATA_ROOT / "trained_weights"

# ============================================================
# Label / submission files
# ============================================================
TRAIN_LABELS_CSV = DATA_ROOT / "train_labels.csv"
SUBMISSION_CSV = DATA_ROOT / "submission.csv"
VAL_SUBMISSION_CSV = DATA_ROOT / "validationsubmission.csv"

# ============================================================
# Derived folders for this pipeline
# ============================================================
ROI_ROOT = DATA_ROOT / "roi_data"
ROI_BOX_DEBUG_DIR = ROI_ROOT / "roi_box_debug"

ROI_TRAIN_DIR = ROI_ROOT / "train"
ROI_VAL_DIR = ROI_ROOT / "validation"
ROI_TEST_DIR = ROI_ROOT / "test"
ROI_UNLABELED_DIR = ROI_ROOT / "unlabeled"

SPLITS_DIR = DATA_ROOT / "splits"
PSEUDO_DIR = DATA_ROOT / "pseudo_labels"

# ============================================================
# Fixed image / ROI settings
# ============================================================
ROI_SIZE = 224
NUM_ROIS = 4

# These are kept fixed from your tuned detector
BRIGHT_ENHANCE_SIGMA = 15
DOT_THRESHOLD = 30

DOT_MIN_W = 1
DOT_MAX_W = 12
DOT_MIN_H = 1
DOT_MAX_H = 12
DOT_MIN_AREA = 3

MERGE_DILATE_KERNEL = 9
MERGE_CLOSE_KERNEL = 15

PANEL_MIN_W = 40
PANEL_MIN_H = 30
PANEL_MIN_ASPECT = 1.0
PANEL_MAX_ASPECT = 2.5
PANEL_MIN_AREA_FRAC = 0.001
PANEL_MAX_AREA_FRAC = 0.08

OVERLAP_IOU_THRESHOLD = 0.2

EXPAND_PAD_X = 35
EXPAND_PAD_Y = 35

EXPECTED_BOXES_PER_HALF = 2
EXPECTED_TOTAL_BOXES = 4

# ============================================================
# Fallback ROI template
# These will be computed later from detected boxes and saved.
# ============================================================
FALLBACK_TEMPLATE_PATH = ROI_ROOT / "fallback_template_boxes.npy"

# ============================================================
# Training defaults
# ============================================================
SEED = 42
NUM_FOLDS = 5
BATCH_SIZE_SSL = 64
BATCH_SIZE_SUP = 16
NUM_WORKERS = 8

SSL_EPOCHS = 100
SUP_EPOCHS = 50

LR_SSL = 1e-3
LR_SUP = 1e-4

# ============================================================
# Utility
# ============================================================
def ensure_project_dirs():
    dirs = [
        LOG_DIR,
        WEIGHTS_DIR,
        ROI_ROOT,
        ROI_BOX_DEBUG_DIR,
        ROI_TRAIN_DIR,
        ROI_VAL_DIR,
        ROI_TEST_DIR,
        ROI_UNLABELED_DIR,
        SPLITS_DIR,
        PSEUDO_DIR,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    ensure_project_dirs()
    print("Created/verified project directories.")