from pathlib import Path
import torch
from typing import Union, Optional, Dict, Any
import os
import warnings
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
NUM_TEST_SAMPLES_CIFAR10 = 10000 # Standard for CIFAR-10 test set
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
    """
    Get the path for a MAGIC model checkpoint.
    
    Args:
        model_id (int): Model identifier (typically 0 for single model).
        step_or_epoch (int): Training step or epoch number.
        
    Returns:
        Path: Complete path to the checkpoint file.
    """
    return MAGIC_CHECKPOINTS_DIR / f"sd_{model_id}_{step_or_epoch}.pt"

def get_magic_scores_path(target_idx: int) -> Path:
    """
    Get the path for MAGIC influence scores.
    
    Args:
        target_idx (int): Target validation image index.
        
    Returns:
        Path: Complete path to the scores file.
    """
    return MAGIC_SCORES_DIR / f"magic_scores_per_step_val_{target_idx}.pkl"

def get_lds_subset_model_checkpoint_path(model_id: int, step_or_epoch: int) -> Path:
    """
    Get the path for an LDS subset model checkpoint.
    
    Args:
        model_id (int): LDS model identifier.
        step_or_epoch (int): Training step or epoch number.
        
    Returns:
        Path: Complete path to the checkpoint file.
    """
    return LDS_CHECKPOINTS_DIR / f"sd_lds_{model_id}_{step_or_epoch}.pt"

def get_lds_model_val_loss_path(model_id: int) -> Path:
    """
    Get the path for LDS model validation losses.
    
    Args:
        model_id (int): LDS model identifier.
        
    Returns:
        Path: Complete path to the validation loss file.
    """
    return LDS_LOSSES_DIR / f"loss_lds_{model_id}.pkl"

# --- MAGIC Analyzer Specific Configurations ---
MAGIC_TARGET_VAL_IMAGE_IDX = 21
MAGIC_NUM_INFLUENTIAL_IMAGES_TO_SHOW = 8

# === SHARED TRAINING HYPERPARAMETERS ===
# These are used by both MAGIC and LDS for consistency
# Aligned with magic.py effective training settings for the model whose checkpoints are replayed.
MODEL_TRAIN_LR = 1e-3
MODEL_TRAIN_EPOCHS = 6
MODEL_TRAIN_BATCH_SIZE = 1000
MODEL_TRAIN_MOMENTUM = 0.875  # magic.py optimizer effectively had 0 momentum
MODEL_TRAIN_WEIGHT_DECAY = 0.001  # magic.py optimizer effectively had 0 WD

# === LEARNING RATE SCHEDULER CONFIGURATION ===
# Options: None, 'StepLR', 'CosineAnnealingLR', 'OneCycleLR'
LR_SCHEDULE_TYPE = None  

# StepLR parameters
STEPLR_STEP_SIZE = 10  # Step size for StepLR
STEPLR_GAMMA = 0.1     # Multiplicative factor for StepLR

# CosineAnnealingLR parameters  
COSINE_T_MAX = None  # Will be auto-calculated as total_steps if None

# OneCycleLR parameters
ONECYCLE_MAX_LR = 1e-2           # Maximum learning rate (higher than base LR)
ONECYCLE_PCT_START = 0.3         # Percentage of cycle spent increasing LR
ONECYCLE_ANNEAL_STRATEGY = 'cos' # 'cos' or 'linear'
ONECYCLE_DIV_FACTOR = 25.0       # initial_lr = max_lr / div_factor
ONECYCLE_FINAL_DIV_FACTOR = 1e4  # min_lr = initial_lr / final_div_factor

# --- Numerical Stability ---
EPSILON_FOR_WEIGHTED_LOSS = 1e-8 # Small epsilon for division in weighted loss calculations

# --- LDS Validator Specific Configurations ---
LDS_TARGET_VAL_IMAGE_IDX_FOR_CORRELATION = MAGIC_TARGET_VAL_IMAGE_IDX # Should match MAGIC_TARGET_VAL_IMAGE_IDX

# Parameters for LDS subset generation
LDS_SUBSET_FRACTION = 0.99
LDS_NUM_SUBSETS_TO_GENERATE = 128 # Max number of subset definitions to generate
LDS_NUM_MODELS_TO_TRAIN = 128    # Number of these subsets to actually train models on

# LDS training uses the same hyperparameters as MAGIC (config.MODEL_TRAIN_*)

# Path to the MAGIC scores file that LDS validator will use
# This should point to the output of magic_analyzer (per-step scores)
MAGIC_SCORES_FILE_FOR_LDS_INPUT = get_magic_scores_path(LDS_TARGET_VAL_IMAGE_IDX_FOR_CORRELATION)

# --- Deterministic Component Instance IDs (for shared components) ---
# Used to ensure identical seed derivation for components shared across analyses
SHARED_MODEL_INSTANCE_ID = "core_training_model_config"
SHARED_DATALOADER_INSTANCE_ID = "core_training_dataloader_config"
SHARED_OPTIMIZER_INSTANCE_ID = "core_training_optimizer_config"
SHARED_SCHEDULER_INSTANCE_ID = "core_training_scheduler_config"

# Inline Validations Control
PERFORM_INLINE_VALIDATIONS = os.getenv('PERFORM_INLINE_VALIDATIONS', 'False').lower() == 'true'
# Set to True (e.g., via environment variable PERFORM_INLINE_VALIDATIONS=True) to enable detailed checks
# during execution. These checks can be resource-intensive and are typically disabled for standard runs.

def validate_config() -> None:
    """
    Validates configuration parameters for consistency and compatibility.
    Raises ValueError if configuration is invalid.
    """
    # Check target image indices alignment
    if MAGIC_TARGET_VAL_IMAGE_IDX != LDS_TARGET_VAL_IMAGE_IDX_FOR_CORRELATION:
        warnings.warn(
            f"MAGIC target image index ({MAGIC_TARGET_VAL_IMAGE_IDX}) differs from "
            f"LDS target image index ({LDS_TARGET_VAL_IMAGE_IDX_FOR_CORRELATION}). "
            f"This may cause LDS validation to fail or use incorrect MAGIC scores."
        )
    
    # Validate target indices are non-negative
    if MAGIC_TARGET_VAL_IMAGE_IDX < 0 or MAGIC_TARGET_VAL_IMAGE_IDX >= NUM_TEST_SAMPLES_CIFAR10:
        raise ValueError(f"MAGIC_TARGET_VAL_IMAGE_IDX ({MAGIC_TARGET_VAL_IMAGE_IDX}) is out of bounds for CIFAR-10 test set (0 to {NUM_TEST_SAMPLES_CIFAR10-1}).")
    
    if LDS_TARGET_VAL_IMAGE_IDX_FOR_CORRELATION < 0 or LDS_TARGET_VAL_IMAGE_IDX_FOR_CORRELATION >= NUM_TEST_SAMPLES_CIFAR10:
        raise ValueError(f"LDS_TARGET_VAL_IMAGE_IDX_FOR_CORRELATION ({LDS_TARGET_VAL_IMAGE_IDX_FOR_CORRELATION}) is out of bounds for CIFAR-10 test set (0 to {NUM_TEST_SAMPLES_CIFAR10-1}).")
    
    # Validate hyperparameters
    if MODEL_TRAIN_LR <= 0:
        raise ValueError(f"MODEL_TRAIN_LR must be positive, got {MODEL_TRAIN_LR}")
    
    if MODEL_TRAIN_EPOCHS <= 0: # Unified check for epochs
        raise ValueError(f"MODEL_TRAIN_EPOCHS must be positive, got {MODEL_TRAIN_EPOCHS}")

    if MODEL_TRAIN_BATCH_SIZE <= 0: # Unified check for batch size
        raise ValueError(f"MODEL_TRAIN_BATCH_SIZE must be positive, got {MODEL_TRAIN_BATCH_SIZE}")
    
    # Check if sufficient training samples
    min_subset_size = int(NUM_TRAIN_SAMPLES * LDS_SUBSET_FRACTION)
    if min_subset_size < MODEL_TRAIN_BATCH_SIZE:
        warnings.warn(
            f"LDS subset size ({min_subset_size}) is smaller than batch size "
            f"({MODEL_TRAIN_BATCH_SIZE}). This may cause training issues."
        )
    
    # Check that subset size won't be larger than total samples
    if min_subset_size > NUM_TRAIN_SAMPLES:
        raise ValueError(f"LDS subset size ({min_subset_size}) cannot exceed total training samples ({NUM_TRAIN_SAMPLES})")
    
    # Validate paths
    if not CIFAR_ROOT:
        raise ValueError("CIFAR_ROOT cannot be empty")
    
    # Validate number of workers
    if DATALOADER_NUM_WORKERS < 0:
        raise ValueError(f"DATALOADER_NUM_WORKERS must be non-negative, got {DATALOADER_NUM_WORKERS}")
    
    # Check device availability
    if DEVICE.type == 'cuda' and not torch.cuda.is_available():
        warnings.warn("CUDA device specified but not available. Using CPU instead.")


def get_config_summary() -> str:
    """
    Returns a summary of key configuration parameters.
    """
    scheduler_info = f"LR Schedule: {LR_SCHEDULE_TYPE}"
    if LR_SCHEDULE_TYPE == 'StepLR':
        scheduler_info += f" (step_size={STEPLR_STEP_SIZE}, gamma={STEPLR_GAMMA})"
    elif LR_SCHEDULE_TYPE == 'CosineAnnealingLR':
        scheduler_info += f" (T_max={COSINE_T_MAX or 'auto'})"
    elif LR_SCHEDULE_TYPE == 'OneCycleLR':
        scheduler_info += f" (max_lr={ONECYCLE_MAX_LR}, pct_start={ONECYCLE_PCT_START})"
    
    return f"""
Configuration Summary:
=====================
Device: {DEVICE}
Seed: {SEED}
CIFAR Root: {CIFAR_ROOT}

=== SHARED TRAINING CONFIGURATION ===
- Training epochs: {MODEL_TRAIN_EPOCHS}
- Batch size: {MODEL_TRAIN_BATCH_SIZE}
- Learning rate: {MODEL_TRAIN_LR}
- Momentum: {MODEL_TRAIN_MOMENTUM}
- Weight decay: {MODEL_TRAIN_WEIGHT_DECAY}
- {scheduler_info}

MAGIC Configuration:
- Target image index: {MAGIC_TARGET_VAL_IMAGE_IDX}
- Replay learning rate: {MODEL_TRAIN_LR} (same as training LR)

LDS Configuration:
- Target image index: {LDS_TARGET_VAL_IMAGE_IDX_FOR_CORRELATION}
- Subset fraction: {LDS_SUBSET_FRACTION}
- Number of subsets: {LDS_NUM_SUBSETS_TO_GENERATE}
- Models to train: {LDS_NUM_MODELS_TO_TRAIN}

Output Directories:
- MAGIC checkpoints: {MAGIC_CHECKPOINTS_DIR}
- MAGIC scores: {MAGIC_SCORES_DIR}
- LDS checkpoints: {LDS_CHECKPOINTS_DIR}
- LDS losses: {LDS_LOSSES_DIR}
"""

# Note: The ensure_output_dirs_exist() function was removed from here.
# It should be called from main_runner.py or relevant class constructors. 

def validate_environment() -> Dict[str, Any]:
    """
    Validates the runtime environment and returns system information.
    
    Returns:
        Dict[str, Any]: Environment information including device, memory, etc.
        
    Raises:
        EnvironmentError: If critical environment requirements are not met.
    """
    env_info = {}
    
    # Check Python version
    import sys
    env_info['python_version'] = sys.version
    if sys.version_info < (3, 8):
        raise EnvironmentError(f"Python 3.8+ required, found {sys.version_info}")
    
    # Check PyTorch version and CUDA availability
    env_info['torch_version'] = torch.__version__
    env_info['cuda_available'] = torch.cuda.is_available()
    
    if torch.cuda.is_available():
        env_info['cuda_device_count'] = torch.cuda.device_count()
        env_info['cuda_version'] = torch.version.cuda
        
        # Check GPU memory
        try:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            env_info['gpu_memory_gb'] = gpu_memory
            if gpu_memory < 2.0:  # Less than 2GB
                warnings.warn(f"Low GPU memory ({gpu_memory:.1f}GB). Consider using --memory_efficient mode.")
        except Exception:
            env_info['gpu_memory_gb'] = 'unknown'
    
    # Check available disk space for outputs
    try:
        if OUTPUTS_DIR.exists():
            statvfs = os.statvfs(OUTPUTS_DIR)
            free_space_gb = (statvfs.f_frsize * statvfs.f_bavail) / 1e9
            env_info['free_disk_space_gb'] = free_space_gb
            if free_space_gb < 1.0:  # Less than 1GB
                warnings.warn(f"Low disk space ({free_space_gb:.1f}GB) in outputs directory.")
    except Exception:
        env_info['free_disk_space_gb'] = 'unknown'
    
    # Check write permissions
    try:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        test_file = OUTPUTS_DIR / ".write_test"
        test_file.write_text("test")
        test_file.unlink()
        env_info['outputs_writable'] = True
    except Exception as e:
        env_info['outputs_writable'] = False
        raise EnvironmentError(f"Cannot write to outputs directory {OUTPUTS_DIR}: {e}")
    
    return env_info 