from pathlib import Path

# --- Project Structure ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent # Assumes src is one level down
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DATA_DIR = PROJECT_ROOT / "data" # If you decide to store persistent data

# --- Common Paths ---
CIFAR_ROOT = Path("/tmp/cifar/") # Or use DATA_DIR / "cifar10" if downloaded locally

# Checkpoints and scores directories for MAGIC (influence analysis)
MAGIC_CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints_magic"
MAGIC_SCORES_DIR = OUTPUTS_DIR / "scores_magic"
MAGIC_PLOTS_DIR = OUTPUTS_DIR / "plots_magic"

# Checkpoints, losses, and scores directories for LDS (subset validation)
LDS_CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints_lds"
LDS_LOSSES_DIR = OUTPUTS_DIR / "losses_lds"
LDS_INDICES_FILE = OUTPUTS_DIR / "indices_lds.pkl" # For saving/loading subset indices
LDS_PLOTS_DIR = OUTPUTS_DIR / "plots_lds"


# --- General Settings ---
RANDOM_SEED = 42
NUM_CLASSES = 10 # For CIFAR-10
NUM_TRAINING_SAMPLES_CIFAR10 = 50000
NUM_TEST_SAMPLES_CIFAR10 = 10000

# --- MAGIC Analyzer (Influence Calculation) Specific Configs ---
MAGIC_TARGET_VAL_IMAGE_IDX = 21
MAGIC_NUM_INFLUENTIAL_IMAGES_TO_SHOW = 8
MAGIC_REPLAY_LEARNING_RATE = 1e-3 # Learning rate for the REPLAY algorithm's approximation

# Training parameters for the model whose influence is being analyzed
MAGIC_TRAIN_BATCH_SIZE = 512
MAGIC_TRAIN_EPOCHS = 2 # Results in ~196 steps for CIFAR10/batch 512
MAGIC_BASE_LR = 0.01 # Optimizer LR for training the main model
MAGIC_MOMENTUM = 0.9
MAGIC_WEIGHT_DECAY = 5e-4


# --- LDS Validator (Subset Training & Correlation) Specific Configs ---
LDS_NUM_MODELS_TO_TRAIN = 32 # Number of models to train on subsets
LDS_SUBSET_SIZE_FRACTION = 0.99 # Fraction of training data for each subset
LDS_NUM_SUBSETS = 128 # Number of subset definitions to generate (can be > LDS_NUM_MODELS_TO_TRAIN)
LDS_TARGET_VAL_IMAGE_IDX_FOR_CORRELATION = 21 # Which validation image's margin to correlate

# Training parameters for subset models (can be same or different from MAGIC model)
LDS_TRAIN_BATCH_SIZE = 512
LDS_TRAIN_EPOCHS = 2 # Epochs for each subset model
LDS_BASE_LR = 0.01
LDS_MOMENTUM = 0.9
LDS_WEIGHT_DECAY = 5e-4

# --- Ensure output directories exist ---
def ensure_output_dirs_exist():
    dirs_to_create = [
        OUTPUTS_DIR, DATA_DIR,
        MAGIC_CHECKPOINTS_DIR, MAGIC_SCORES_DIR, MAGIC_PLOTS_DIR,
        LDS_CHECKPOINTS_DIR, LDS_LOSSES_DIR, LDS_PLOTS_DIR
    ]
    for d in dirs_to_create:
        d.mkdir(parents=True, exist_ok=True)

# Call it once if this module is imported to create dirs,
# or call explicitly from main runner.
# ensure_output_dirs_exist() 