from pathlib import Path
import torch
# import os # Not strictly necessary if pathlib is used for all path ops

# --- Project Structure ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
# DATA_DIR = PROJECT_ROOT / "data" # Uncomment if you have a persistent local data folder

# --- General Settings ---
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 10
NUM_TRAIN_SAMPLES = 50000 # Standardized name for CIFAR-10 training samples
# NUM_TEST_SAMPLES_CIFAR10 = 10000 # If needed elsewhere

# --- Data Handling ---
CIFAR_ROOT = '/tmp/cifar/' # Default path for CIFAR-10 download
DATALOADER_NUM_WORKERS = 8 # Default from magic.py/lds.py

# --- Output Subdirectories (Standardized) ---
# MAGIC Analysis Outputs
MAGIC_CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints_magic"
MAGIC_SCORES_DIR = OUTPUTS_DIR / "scores_magic"
MAGIC_PLOTS_DIR = OUTPUTS_DIR / "plots_magic" # Specific to MAGIC plots
BATCH_DICT_FILE = OUTPUTS_DIR / "magic_batch_dict.pkl" # For saving/loading batch_dict

# LDS Validator Outputs
LDS_CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints_lds"
LDS_LOSSES_DIR = OUTPUTS_DIR / "losses_lds"
LDS_INDICES_FILE = OUTPUTS_DIR / "indices_lds.pkl"
LDS_PLOTS_DIR = OUTPUTS_DIR / "plots_lds" # Specific to LDS plots

# --- File Naming Helper Functions (using Path objects) ---
def get_magic_checkpoint_path(model_id: int, step_or_epoch: int) -> Path:
    return MAGIC_CHECKPOINTS_DIR / f"sd_{model_id}_{step_or_epoch}.pt"

def get_magic_scores_path(target_idx: int) -> Path:
    # This will store per-step scores from magic_analyzer
    return MAGIC_SCORES_DIR / f"magic_scores_per_step_val_{target_idx}.pkl"

def get_lds_subset_model_checkpoint_path(model_id: int, step_or_epoch: int) -> Path:
    return LDS_CHECKPOINTS_DIR / f"sd_lds_{model_id}_{step_or_epoch}.pt"

def get_lds_model_val_loss_path(model_id: int) -> Path:
    return LDS_LOSSES_DIR / f"loss_lds_{model_id}.pkl"

# --- MAGIC Analyzer Specific Configurations ---
MAGIC_TARGET_VAL_IMAGE_IDX = 21
MAGIC_NUM_INFLUENTIAL_IMAGES_TO_SHOW = 8

# Hyperparameters for training the main model in MAGIC analysis (to generate checkpoints for replay)
# Aligned with magic.py effective training settings for the model whose checkpoints are replayed.
MAGIC_MODEL_TRAIN_LR = 1e-3
MAGIC_MODEL_TRAIN_EPOCHS = 2
MAGIC_MODEL_TRAIN_BATCH_SIZE = 512
MAGIC_MODEL_TRAIN_MOMENTUM = 0.0 # magic.py optimizer effectively had 0 momentum
MAGIC_MODEL_TRAIN_WEIGHT_DECAY = 0.0 # magic.py optimizer effectively had 0 WD
MAGIC_MODEL_LABEL_SMOOTHING = 0.0 # magic.py used 0.0 for its CrossEntropyLoss

# Hyperparameters for the REPLAY influence calculation itself
MAGIC_REPLAY_LEARNING_RATE = 1e-3 # lr used in h_t calculation in magic.py

# --- LDS Validator Specific Configurations ---
LDS_TARGET_VAL_IMAGE_IDX_FOR_CORRELATION = MAGIC_TARGET_VAL_IMAGE_IDX # Should match MAGIC_TARGET_VAL_IMAGE_IDX

# Parameters for LDS subset generation
LDS_SUBSET_FRACTION = 0.99
LDS_NUM_SUBSETS_TO_GENERATE = 128 # Max number of subset definitions to generate
LDS_NUM_MODELS_TO_TRAIN = 128    # Number of these subsets to actually train models on

# Hyperparameters for training LDS subset models
# Aligned with lds.py effective training settings.
LDS_MODEL_TRAIN_LR = 1e-3
LDS_MODEL_TRAIN_EPOCHS = 2
LDS_MODEL_TRAIN_BATCH_SIZE = 512
LDS_MODEL_TRAIN_MOMENTUM = 0.0 # LDS_MOMENTUM was set to 0.0 in previous step
LDS_MODEL_TRAIN_WEIGHT_DECAY = 0.0 # LDS_WEIGHT_DECAY was set to 0.0 in previous step
# LDS_MODEL_LABEL_SMOOTHING = 0.0 # lds.py did not use label smoothing for its weighted loss calculation

# Path to the MAGIC scores file that LDS validator will use
# This should point to the output of magic_analyzer (per-step scores)
MAGIC_SCORES_FILE_FOR_LDS_INPUT = get_magic_scores_path(LDS_TARGET_VAL_IMAGE_IDX_FOR_CORRELATION)


# Note: The ensure_output_dirs_exist() function was removed from here.
# It should be called from main_runner.py or relevant class constructors. 