from pathlib import Path

# ==========================
# Project Paths
# ==========================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"

TRAIN_DIR = PROJECT_ROOT / "dataset" / "train"
TEST_DIR = PROJECT_ROOT / "dataset" / "test"

TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"
SUBMISSION_CSV = DATA_DIR / "submission (1).csv"

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
LOG_DIR = PROJECT_ROOT / "logs"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# ==========================
# Training
# ==========================

IMAGE_SIZE = 224

BATCH_SIZE = 16

NUM_WORKERS = 8

EPOCHS = 30

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

RANDOM_SEED = 42

DEVICE = "cuda"

# ==========================
# Model
# ==========================

NUM_LABELS = 10

BEST_THRESHOLDS = [
    0.65,
    0.25,
    0.25,
    0.65,
    0.50,
    0.65,
    0.90,
    0.65,
    0.65,
    0.85,
]