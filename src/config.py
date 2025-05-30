from pathlib import Path
import torch
from typing import Union, Optional, Dict, Any, Callable, List
import os
import warnings

from .model_def import construct_resnet9_paper, construct_rn9, ResNet9TableArch, make_airbench94_adapted

# Import run management and path functions from the new run_manager module
from . import run_manager 
# Make key run management functions available if they were directly imported elsewhere, e.g., in main_runner
# Alternatively, other modules should import them from run_manager directly.
# For now, let's re-export a few common ones for smoother transition.
init_run_directory = run_manager.init_run_directory
get_current_run_dir = run_manager.get_current_run_dir
list_runs = run_manager.list_runs
get_latest_run_id = run_manager.get_latest_run_id
clean_magic_checkpoints = run_manager.clean_magic_checkpoints
clean_lds_checkpoints = run_manager.clean_lds_checkpoints
get_run_size_info = run_manager.get_run_size_info
save_run_metadata = run_manager.save_run_metadata # save_run_metadata will use get_current_config_dict from this file
clean_run_checkpoints = run_manager.clean_run_checkpoints

# Re-export path generation functions
get_magic_checkpoints_dir = run_manager.get_magic_checkpoints_dir
get_magic_scores_dir = run_manager.get_magic_scores_dir
get_magic_plots_dir = run_manager.get_magic_plots_dir
get_magic_logs_dir = run_manager.get_magic_logs_dir
get_batch_dict_file = run_manager.get_batch_dict_file
get_lds_checkpoints_dir = run_manager.get_lds_checkpoints_dir
get_lds_losses_dir = run_manager.get_lds_losses_dir
get_lds_indices_file = run_manager.get_lds_indices_file
get_lds_plots_dir = run_manager.get_lds_plots_dir
get_lds_logs_dir = run_manager.get_lds_logs_dir
get_magic_checkpoint_path = run_manager.get_magic_checkpoint_path
get_magic_scores_path = run_manager.get_magic_scores_path
get_lds_subset_model_checkpoint_path = run_manager.get_lds_subset_model_checkpoint_path
get_lds_model_val_loss_path = run_manager.get_lds_model_val_loss_path
get_magic_training_log_path = run_manager.get_magic_training_log_path
get_magic_replay_log_path = run_manager.get_magic_replay_log_path
get_lds_training_log_path = run_manager.get_lds_training_log_path
get_magic_scores_file_for_lds_input = run_manager.get_magic_scores_file_for_lds_input


# --- Project Structure --- (Keep constants like PROJECT_ROOT and OUTPUTS_DIR here)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# OUTPUTS_DIR = PROJECT_ROOT / "outputs" # Old definition
OUTPUTS_DIR_DEFAULT = PROJECT_ROOT / "outputs"
OUTPUTS_DIR = Path(os.getenv('REPLAY_OUTPUTS_DIR', OUTPUTS_DIR_DEFAULT))

# DATA_DIR = PROJECT_ROOT / "data" # Uncomment if you have a persistent local data folder

# --- General Settings --- (Keep these core settings)
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ... (rest of the config file from line 290 onwards, starting with NUM_CLASSES = 10)
# Ensure get_current_config_dict remains here, as save_run_metadata (now in run_manager)
# will call project_config.get_current_config_dict()
NUM_CLASSES = 10
NUM_TRAIN_SAMPLES = 50000 # Standardized name for CIFAR-10 training samples
NUM_TEST_SAMPLES_CIFAR10 = 10000 # Standard for CIFAR-10 test set
# NUM_TEST_SAMPLES_CIFAR10 = 10000 # If needed elsewhere

# --- Data Handling ---
CIFAR_ROOT = '/tmp/cifar/' # Default path for CIFAR-10 download
DATALOADER_NUM_WORKERS = 8 # Default from magic.py/lds.py

# --- Output Subdirectories (Standardized) ---
# These are now dynamic based on the current run directory and handled by run_manager.py
# def _get_output_dir(subdir: str) -> Path:
#     """Get output directory path within current run."""
#     return get_current_run_dir() / subdir

# MAGIC Analysis Outputs (Handled by run_manager.py)
# def get_magic_checkpoints_dir() -> Path:
# ... all path getter functions removed as they are now in run_manager.py and re-exported ...

# --- MAGIC Analyzer Specific Configurations ---
MAGIC_TARGET_VAL_IMAGE_IDX = 21 # QUICK TEST: Use index 21
MAGIC_NUM_INFLUENTIAL_IMAGES_TO_SHOW = 8

# Clipping options for MAGIC Replay
MAGIC_REPLAY_ENABLE_GRAD_CLIPPING = False       # Enable/disable gradient clipping during replay
MAGIC_REPLAY_MAX_GRAD_NORM = 0.5               # Max norm for gradients if clipping is enabled
MAGIC_REPLAY_ENABLE_PARAM_CLIPPING = False      # Enable/disable parameter norm warning and hard clipping
MAGIC_REPLAY_MAX_PARAM_NORM_WARNING = 5.0      # Threshold for parameter norm warnings if enabled
MAGIC_REPLAY_PARAM_CLIP_NORM_HARD = 10.0       # Threshold for hard parameter norm clipping if enabled

# === PAPER SPECIFICATION: 50 MEASUREMENT FUNCTIONS ===
# According to the paper: "we consider 50 measurement functions φi corresponding to 
# loss on 50 different CIFAR-10 test samples"
PAPER_NUM_MEASUREMENT_FUNCTIONS = 50
PAPER_MEASUREMENT_TARGET_INDICES = list(range(PAPER_NUM_MEASUREMENT_FUNCTIONS))  # [0, 1, 2, ..., 49]

# For compatibility with existing single-target analysis, we can use the first target
# or specify a particular target from the 50 measurement functions
# MAGIC_TARGET_VAL_IMAGE_IDX_FROM_PAPER = PAPER_MEASUREMENT_TARGET_INDICES[21]  # Use index 21 as default

# === SHARED TRAINING HYPERPARAMETERS ===
# These are used by both MAGIC and LDS for consistency
# Updated to match ResNet-9 on CIFAR-10 specifications from [Jor24a]
# and user-provided large-batch SGD recipe.
MODEL_TRAIN_LR = 1e-3  # Peak LR for OneCycleLR.
MODEL_TRAIN_EPOCHS = 1 # Reduced for faster integration testing
WARMUP_EPOCHS = 0       # OneCycleLR handles its own warmup via pct_start. Set to 0.
MODEL_TRAIN_BATCH_SIZE = 256
MODEL_TRAIN_MOMENTUM = 0
MODEL_TRAIN_WEIGHT_DECAY = 0 # For parameters subject to decay
MODEL_TRAIN_NESTEROV = False # Common with momentum, though not explicitly in table. User had it.

# Optimizer Type ('SGD' or 'Adam')
MODEL_TRAIN_OPTIMIZER = 'SGD'  # Reverted analyzer primarily supports SGD. Adam settings below are not used by it.

# Adam Specific Hyperparameters (only used if MODEL_TRAIN_OPTIMIZER is 'Adam')
# Note: These are NOT used by the reverted src/magic_analyzer.py which uses manual SGD.
# If not specified here, sensible defaults will be used by the Adam optimizer itself.
# Using MODEL_TRAIN_LR as the default learning rate for Adam as well.
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPSILON = 1e-8
# ADAM_WEIGHT_DECAY is typically handled by the optimizer, MODEL_TRAIN_WEIGHT_DECAY can be used.

# === MODEL ARCHITECTURE CONFIGURATION ===
# Allows for switching the model architecture by changing this function reference.
# MODEL_CREATOR_FUNCTION: Callable[..., torch.nn.Module] = make_airbench94_adapted
# To use the ResNet-9 based on the user's table interpretation:
# MODEL_CREATOR_FUNCTION: Callable[..., torch.nn.Module] = ResNet9TableArch
# To use the original ResNet-9 paper implementation from this project:
# MODEL_CREATOR_FUNCTION: Callable[..., torch.nn.Module] = construct_resnet9_paper
# To use the alternative ResNet-9 implementation:
MODEL_CREATOR_FUNCTION: Callable[..., torch.nn.Module] = construct_rn9

# Store the alternative for reference or programmatic switching if needed elsewhere
# ALTERNATIVE_MODEL_CREATOR_FUNCTION: Callable[..., torch.nn.Module] = construct_rn9

# === RESNET-9 ARCHITECTURE PARAMETERS ===
# Architecture-specific parameters from the paper (primarily for construct_resnet9_paper)
RESNET9_WIDTH_MULTIPLIER = 2.5    # Width multiplier for channels
RESNET9_BIAS_SCALE = 1.0          # Bias scale for initialization AND LR scaling for biases/BN (reduced to 1.0 for numerical stability in manual replay)
RESNET9_FINAL_LAYER_SCALE = 0.04  # Final layer scaling factor
RESNET9_POOLING_EPSILON = 0.1     # Epsilon for log-sum-exp pooling

# === LEARNING RATE SCHEDULER CONFIGURATION ===
# Options: None, 'StepLR', 'CosineAnnealingLR', 'OneCycleLR'
# Note: If WARMUP_EPOCHS > 0, the main scheduler (e.g., CosineAnnealingLR)
# typically starts after WARMUP_EPOCHS. The training script should handle the
# warmup phase (e.g., linear ramp-up) and then transition to this scheduler.
LR_SCHEDULE_TYPE = None

# StepLR parameters
STEPLR_STEP_SIZE = 10 # Not used if OneCycleLR is active
STEPLR_GAMMA = 0.1    # Not used if OneCycleLR is active

# CosineAnnealingLR parameters
# COSINE_T_MAX is dynamically determined in the training script based on:
# (MODEL_TRAIN_EPOCHS - WARMUP_EPOCHS) * steps_per_epoch.
COSINE_T_MAX = None # Not used if OneCycleLR is active

# OneCycleLR parameters (not used if LR_SCHEDULE_TYPE is not 'OneCycleLR')
ONECYCLE_MAX_LR = MODEL_TRAIN_LR # Base max_lr. Optimizer groups will scale this.
ONECYCLE_PCT_START = 0.5 # Corresponds to LR peak time from table
ONECYCLE_ANNEAL_STRATEGY = 'linear' # From "One-cycle Linear" in table
ONECYCLE_DIV_FACTOR = 1.0 / 0.07 # max_lr / initial_lr; initial_lr = 0.07 * max_lr

# Corrected calculation for final_div_factor:
# Table 1 specifies: final_lr = max_lr * 0.2 = 1.2 * 0.2 = 0.24
# PyTorch formula: min_lr = initial_lr / final_div_factor
# initial_lr = max_lr / div_factor = 1.2 / (1/0.07) = 1.2 * 0.07 = 0.084
# Therefore: final_div_factor = initial_lr / target_final_lr = 0.084 / 0.24 = 0.35
ONECYCLE_FINAL_DIV_FACTOR = 0.35  # Achieves final_lr = 0.084 / 0.35 = 0.24 = max_lr * 0.2

# --- Numerical Stability ---
EPSILON_FOR_WEIGHTED_LOSS = 1e-8

# --- LDS Validator Specific Configurations ---
LDS_TARGET_VAL_IMAGE_IDX_FOR_CORRELATION = MAGIC_TARGET_VAL_IMAGE_IDX # Should match MAGIC_TARGET_VAL_IMAGE_IDX
LDS_INDICES_FILE = "indices_lds.pkl" # Name of the file storing LDS subset indices

# Parameters for LDS subset generation
LDS_SUBSET_FRACTION = 0.99
LDS_NUM_SUBSETS_TO_GENERATE = 1 # Reduced to save disk space and time
LDS_NUM_MODELS_TO_TRAIN = 1    # Reduced to save disk space and time

# LDS training uses the same hyperparameters as MAGIC (config.MODEL_TRAIN_*) 

# Path to the MAGIC scores file that LDS validator will use
# This should point to the output of magic_analyzer (per-step scores)
# This function is now re-exported from run_manager
# def get_magic_scores_file_for_lds_input() -> Path:
#     """Get the MAGIC scores file path that LDS will use."""
#     return get_magic_scores_path(LDS_TARGET_VAL_IMAGE_IDX_FOR_CORRELATION)

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

# --- Enhanced Configuration Validation ---

def validate_training_compatibility() -> None:
    """
    Validates training configuration for compatibility and resource requirements.
    Raises ValueError if configuration is incompatible.
    """
    # Check batch size vs available GPU memory
    if DEVICE.type == 'cuda' and torch.cuda.is_available():
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        estimated_memory_per_sample = 0.01  # Rough estimate in GB
        estimated_batch_memory = MODEL_TRAIN_BATCH_SIZE * estimated_memory_per_sample
        
        if estimated_batch_memory > gpu_memory_gb * 0.8:  # Leave 20% headroom
            warnings.warn(
                f"Large batch size ({MODEL_TRAIN_BATCH_SIZE}) may exceed GPU memory "
                f"({gpu_memory_gb:.1f}GB). Consider reducing batch size."
            )
    
    # Check learning rate scheduler compatibility
    if LR_SCHEDULE_TYPE == 'OneCycleLR':
        if ONECYCLE_PCT_START <= 0 or ONECYCLE_PCT_START >= 1:
            raise ValueError(f"ONECYCLE_PCT_START must be between 0 and 1, got {ONECYCLE_PCT_START}")
        
        if ONECYCLE_DIV_FACTOR <= 1:
            raise ValueError(f"ONECYCLE_DIV_FACTOR must be > 1, got {ONECYCLE_DIV_FACTOR}")
        
        if ONECYCLE_FINAL_DIV_FACTOR <= 0:
            raise ValueError(f"ONECYCLE_FINAL_DIV_FACTOR must be > 0, got {ONECYCLE_FINAL_DIV_FACTOR}")
    
    # Check model architecture parameters
    if RESNET9_WIDTH_MULTIPLIER <= 0:
        raise ValueError(f"RESNET9_WIDTH_MULTIPLIER must be positive, got {RESNET9_WIDTH_MULTIPLIER}")
    
    if RESNET9_BIAS_SCALE <= 0:
        raise ValueError(f"RESNET9_BIAS_SCALE must be positive, got {RESNET9_BIAS_SCALE}")
    
    # Check LDS configuration
    if LDS_SUBSET_FRACTION <= 0 or LDS_SUBSET_FRACTION > 1:
        raise ValueError(f"LDS_SUBSET_FRACTION must be between 0 and 1, got {LDS_SUBSET_FRACTION}")
    
    min_subset_size = int(NUM_TRAIN_SAMPLES * LDS_SUBSET_FRACTION)
    if min_subset_size < MODEL_TRAIN_BATCH_SIZE * 2:  # Need at least 2 batches
        raise ValueError(
            f"LDS subset size ({min_subset_size}) too small for batch size ({MODEL_TRAIN_BATCH_SIZE}). "
            f"Increase LDS_SUBSET_FRACTION or decrease MODEL_TRAIN_BATCH_SIZE."
        )

def validate_config() -> None:
    """
    Validates configuration parameters for consistency and compatibility.
    Raises ValueError if configuration is invalid.
    """
    # Original validations
    if MAGIC_TARGET_VAL_IMAGE_IDX != LDS_TARGET_VAL_IMAGE_IDX_FOR_CORRELATION:
        warnings.warn(
            f"MAGIC target image index ({MAGIC_TARGET_VAL_IMAGE_IDX}) differs from "
            f"LDS target image index ({LDS_TARGET_VAL_IMAGE_IDX_FOR_CORRELATION}). "
            f"This may cause LDS validation to fail or use incorrect MAGIC scores."
        )
    
    if MAGIC_TARGET_VAL_IMAGE_IDX < 0 or MAGIC_TARGET_VAL_IMAGE_IDX >= NUM_TEST_SAMPLES_CIFAR10:
        raise ValueError(f"MAGIC_TARGET_VAL_IMAGE_IDX ({MAGIC_TARGET_VAL_IMAGE_IDX}) is out of bounds for CIFAR-10 test set (0 to {NUM_TEST_SAMPLES_CIFAR10-1}).")
    
    if LDS_TARGET_VAL_IMAGE_IDX_FOR_CORRELATION < 0 or LDS_TARGET_VAL_IMAGE_IDX_FOR_CORRELATION >= NUM_TEST_SAMPLES_CIFAR10:
        raise ValueError(f"LDS_TARGET_VAL_IMAGE_IDX_FOR_CORRELATION ({LDS_TARGET_VAL_IMAGE_IDX_FOR_CORRELATION}) is out of bounds for CIFAR-10 test set (0 to {NUM_TEST_SAMPLES_CIFAR10-1}).")
    
    if MODEL_TRAIN_LR <= 0:
        raise ValueError(f"MODEL_TRAIN_LR must be positive, got {MODEL_TRAIN_LR}")
    
    if MODEL_TRAIN_EPOCHS <= 0:
        raise ValueError(f"MODEL_TRAIN_EPOCHS must be positive, got {MODEL_TRAIN_EPOCHS}")

    if MODEL_TRAIN_BATCH_SIZE <= 0:
        raise ValueError(f"MODEL_TRAIN_BATCH_SIZE must be positive, got {MODEL_TRAIN_BATCH_SIZE}")
    
    if DATALOADER_NUM_WORKERS < 0:
        raise ValueError(f"DATALOADER_NUM_WORKERS must be non-negative, got {DATALOADER_NUM_WORKERS}")
    
    if not CIFAR_ROOT:
        raise ValueError("CIFAR_ROOT cannot be empty")
    
    # Enhanced validations
    validate_training_compatibility()

def _get_optimizer_summary(config_dict: Dict[str, Any]) -> List[str]:
    """Helper to generate optimizer-specific summary lines."""
    summary = []
    summary.append(f"  MODEL_TRAIN_OPTIMIZER: {config_dict['MODEL_TRAIN_OPTIMIZER']} (Note: Reverted MagicAnalyzer uses manual SGD)")
    if config_dict['MODEL_TRAIN_OPTIMIZER'].lower() == 'sgd':
        summary.append(f"  SGD LR: {config_dict['MODEL_TRAIN_LR']}")
        summary.append(f"  SGD Momentum: {config_dict['MODEL_TRAIN_MOMENTUM']}")
        summary.append(f"  SGD Nesterov: {config_dict['MODEL_TRAIN_NESTEROV']}")
    # The elif for Adam is unlikely to be hit given MODEL_TRAIN_OPTIMIZER is 'SGD'
    # but kept for completeness if config were to change.
    elif config_dict['MODEL_TRAIN_OPTIMIZER'].lower() == 'adam': 
        summary.append(f"  Adam LR (config): {config_dict['MODEL_TRAIN_LR']}") 
        summary.append(f"  Adam Beta1 (config): {config_dict['ADAM_BETA1']}")
        summary.append(f"  Adam Beta2 (config): {config_dict['ADAM_BETA2']}")
        summary.append(f"  Adam Epsilon (config): {config_dict['ADAM_EPSILON']}")
    summary.append(f"  Weight Decay: {config_dict['MODEL_TRAIN_WEIGHT_DECAY']}")
    return summary

def _get_scheduler_summary(config_dict: Dict[str, Any]) -> List[str]:
    """Helper to generate scheduler-specific summary lines."""
    summary = []
    scheduler_specific_info = ""
    if config_dict['LR_SCHEDULE_TYPE'] == 'StepLR':
        scheduler_specific_info = f" (step_size={config_dict['STEPLR_STEP_SIZE']}, gamma={config_dict['STEPLR_GAMMA']})"
    elif config_dict['LR_SCHEDULE_TYPE'] == 'CosineAnnealingLR':
        t_max_val = config_dict['COSINE_T_MAX'] or 'auto (dynamic in train script)'
        scheduler_specific_info = f" (T_max={t_max_val})"
    elif config_dict['LR_SCHEDULE_TYPE'] == 'OneCycleLR':
        scheduler_specific_info = f" (max_lr={config_dict['ONECYCLE_MAX_LR']}, pct_start={config_dict['ONECYCLE_PCT_START']})"

    scheduler_line = f"  LR Scheduler: {config_dict['LR_SCHEDULE_TYPE']}{scheduler_specific_info if config_dict['LR_SCHEDULE_TYPE'] else ''}"
    if config_dict['WARMUP_EPOCHS'] > 0:
        scheduler_line += f", Warmup: {config_dict['WARMUP_EPOCHS']} epochs (linear ramp to MODEL_TRAIN_LR)"
    summary.append(scheduler_line)
    return summary

def get_config_summary() -> str:
    """
    Returns a summary of key configuration parameters.
    """
    config_dict = get_current_config_dict()
    
    summary_lines = ["Configuration Summary:"]
    summary_lines.append(f"  SEED: {config_dict['SEED']}")
    summary_lines.append(f"  DEVICE: {config_dict['DEVICE']}")
    summary_lines.append(f"  MODEL_CREATOR_FUNCTION: {config_dict.get('MODEL_CREATOR_FUNCTION', 'N/A').__name__ if config_dict.get('MODEL_CREATOR_FUNCTION') else 'N/A'}")
    
    summary_lines.extend(_get_optimizer_summary(config_dict))
    
    summary_lines.append(f"  Epochs: {config_dict['MODEL_TRAIN_EPOCHS']}")
    summary_lines.append(f"  Batch Size: {config_dict['MODEL_TRAIN_BATCH_SIZE']}")
    
    # WARMUP_EPOCHS is now part of _get_scheduler_summary logic if relevant
    # summary_lines.append(f"  Warmup Epochs: {config_dict['WARMUP_EPOCHS']}") 
    summary_lines.extend(_get_scheduler_summary(config_dict))
    
    summary_lines.append(f"  MAGIC Target Image Index: {config_dict['MAGIC_TARGET_VAL_IMAGE_IDX']}")
    summary_lines.append(f"  MAGIC Replay Learning Rate: {config_dict['MODEL_TRAIN_LR']} (same as training LR)")
    summary_lines.append(f"  LDS Target Image Index: {config_dict['LDS_TARGET_VAL_IMAGE_IDX_FOR_CORRELATION']}")
    summary_lines.append(f"  LDS Subset Fraction: {config_dict['LDS_SUBSET_FRACTION']}")
    summary_lines.append(f"  LDS Number of Subsets: {config_dict['LDS_NUM_SUBSETS_TO_GENERATE']}")
    summary_lines.append(f"  LDS Models to Train: {config_dict['LDS_NUM_MODELS_TO_TRAIN']}")
    summary_lines.append(f"  MAGIC Scores File for LDS Input: <dynamic based on run context>")
    
    # Use run_manager for these paths now
    summary_lines.append(f"  MAGIC Checkpoints Directory: {run_manager.get_magic_checkpoints_dir()}")
    summary_lines.append(f"  MAGIC Scores Directory: {run_manager.get_magic_scores_dir()}")
    summary_lines.append(f"  LDS Checkpoints Directory: {run_manager.get_lds_checkpoints_dir()}")
    summary_lines.append(f"  LDS Losses Directory: {run_manager.get_lds_losses_dir()}")
    return "\n".join(summary_lines)

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
            if gpu_memory < 4.0:  # Increased warning threshold
                warnings.warn(f"GPU memory ({gpu_memory:.1f}GB) might be low for large batches.")
        except Exception:
            env_info['gpu_memory_gb'] = 'unknown'
    
    # Check available disk space for outputs
    try:
        if OUTPUTS_DIR.exists():
            statvfs = os.statvfs(OUTPUTS_DIR)
            free_space_gb = (statvfs.f_frsize * statvfs.f_bavail) / 1e9
            env_info['free_disk_space_gb'] = free_space_gb
            if free_space_gb < 5.0:  # Increased warning threshold
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

def get_current_config_dict() -> Dict[str, Any]:
    # Helper to get all relevant config vars (globals in this module)
    # Exclude functions and modules
    return {k: v for k, v in globals().items() 
            if not k.startswith('_') and not callable(v) and not isinstance(v, type(Path))} # crude filter 