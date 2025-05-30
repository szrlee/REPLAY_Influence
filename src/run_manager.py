from pathlib import Path
import torch # Keep for potential type hints or future use, though not directly used by moved fns
from typing import Union, Optional, Dict, Any, Callable # Keep for type hints
import os # Keep for os.path.join or other os utils if any are used
import warnings # Keep for warnings

# --- Run Management --- (Moved from config.py)
import datetime
import json
import random
import string

# Global variable to track current run directory
CURRENT_RUN_DIR: Optional[Path] = None
USE_TIMESTAMPED_RUNS = True  # Can be overridden by environment variable
RUNS_DIR_NAME = "runs"  # Subdirectory name for all runs

# Need to define OUTPUTS_DIR here, or pass it to these functions, 
# or import it from config if config still defines it.
# For now, let's assume config.py will still define PROJECT_ROOT and OUTPUTS_DIR
# and we will import them.
from . import config as project_config # Renamed to avoid circular dependency if config imports this

# Also need SEED, DEVICE, MODEL_TRAIN_EPOCHS etc. for _update_runs_registry
# These should also come from project_config

def generate_run_id() -> str:
    """Generate a unique run ID with timestamp and random suffix."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
    return f"{timestamp}_{random_suffix}"

def init_run_directory(run_id: Optional[str] = None, use_existing: bool = False) -> Path:
    """
    Initialize a new run directory or use an existing one.
    
    Args:
        run_id: Specific run ID to use. If None, generates a new one.
        use_existing: If True and run_id is provided, use existing directory.
        
    Returns:
        Path: The run directory path.
    """
    global CURRENT_RUN_DIR
    
    if not USE_TIMESTAMPED_RUNS:
        # Use flat structure (backward compatibility)
        CURRENT_RUN_DIR = project_config.OUTPUTS_DIR
        return CURRENT_RUN_DIR
    
    runs_base = project_config.OUTPUTS_DIR / RUNS_DIR_NAME
    runs_base.mkdir(parents=True, exist_ok=True)
    
    if run_id and use_existing:
        # Use existing run directory
        run_dir = runs_base / run_id
        if not run_dir.exists():
            raise ValueError(f"Run directory {run_dir} does not exist")
        CURRENT_RUN_DIR = run_dir
    else:
        # Create new run directory
        if not run_id:
            run_id = generate_run_id()
        run_dir = runs_base / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        CURRENT_RUN_DIR = run_dir
        
        # Update latest symlink
        latest_link = project_config.OUTPUTS_DIR / "latest"
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        # Symlink target should be relative to the symlink's parent directory
        latest_link.symlink_to(run_dir.relative_to(project_config.OUTPUTS_DIR))
        
        # Add to runs registry
        _update_runs_registry(run_id, "created")
    
    return CURRENT_RUN_DIR

def get_current_run_dir() -> Path:
    """Get the current run directory, initializing if needed."""
    global CURRENT_RUN_DIR
    if CURRENT_RUN_DIR is None:
        init_run_directory() # Will use project_config.OUTPUTS_DIR internally
    return CURRENT_RUN_DIR

def _update_runs_registry(run_id: str, status: str) -> None:
    """Update the runs registry with information about a run."""
    registry_file = project_config.OUTPUTS_DIR / "runs_registry.json"
    
    # Load existing registry
    if registry_file.exists():
        with open(registry_file, 'r') as f:
            registry = json.load(f)
    else:
        registry = {"runs": {}}
    
    # Update registry
    if run_id not in registry["runs"]:
        registry["runs"][run_id] = {}
    
    registry["runs"][run_id].update({
        "status": status,
        "timestamp": datetime.datetime.now().isoformat(),
        "config": {
            "seed": project_config.SEED,
            "device": str(project_config.DEVICE),
            "model_train_epochs": project_config.MODEL_TRAIN_EPOCHS,
            "model_train_batch_size": project_config.MODEL_TRAIN_BATCH_SIZE,
            "magic_target_idx": project_config.MAGIC_TARGET_VAL_IMAGE_IDX,
            "lds_target_idx": project_config.LDS_TARGET_VAL_IMAGE_IDX_FOR_CORRELATION,
        }
    })
    
    # Save registry
    with open(registry_file, 'w') as f:
        json.dump(registry, f, indent=2)

def mark_run_deleted(run_id: str) -> None:
    """Mark a run as 'deleted' in the registry."""
    _update_runs_registry(run_id, "deleted")

def list_runs() -> Dict[str, Any]:
    """List all available runs."""
    registry_file = project_config.OUTPUTS_DIR / "runs_registry.json"
    if not registry_file.exists():
        return {"runs": {}}
    
    with open(registry_file, 'r') as f:
        return json.load(f)

def get_latest_run_id() -> Optional[str]:
    """Get the ID of the latest run."""
    latest_link = project_config.OUTPUTS_DIR / "latest"
    if latest_link.exists() and latest_link.is_symlink():
        # Get the target of the symlink
        target = latest_link.readlink()
        # Extract run ID from path (runs/YYYYMMDD_HHMMSS_xxxxxx)
        if target.parts[0] == RUNS_DIR_NAME and len(target.parts) > 1:
            return target.parts[1]
    
    # Fallback: check registry for most recent
    registry = list_runs()
    if registry["runs"]:
        # Sort by timestamp
        sorted_runs = sorted(
            registry["runs"].items(),
            key=lambda x: x[1].get("timestamp", ""),
            reverse=True
        )
        if sorted_runs:
            return sorted_runs[0][0]
    
    return None

def clean_magic_checkpoints(run_dir: Optional[Path] = None) -> None:
    """
    Clean only MAGIC checkpoint files to save disk space.
    Preserves logs, scores, plots, and other outputs.
    
    Args:
        run_dir: Specific run directory to clean. If None, uses current run.
    """
    if run_dir is None:
        run_dir = get_current_run_dir()
    
    checkpoints_dir = run_dir / "checkpoints_magic"
    if checkpoints_dir.exists() and checkpoints_dir.is_dir():
        import shutil
        shutil.rmtree(checkpoints_dir)
        print(f"Cleaned MAGIC checkpoints from: {checkpoints_dir}")
        
        # Recreate empty directory for future use
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
    else:
        print(f"No MAGIC checkpoints found at: {checkpoints_dir}")

def clean_lds_checkpoints(run_dir: Optional[Path] = None) -> None:
    """
    Clean only LDS checkpoint files to save disk space.
    Preserves logs, losses, plots, and other outputs.
    
    Args:
        run_dir: Specific run directory to clean. If None, uses current run.
    """
    if run_dir is None:
        run_dir = get_current_run_dir()
    
    checkpoints_dir = run_dir / "checkpoints_lds"
    if checkpoints_dir.exists() and checkpoints_dir.is_dir():
        import shutil
        shutil.rmtree(checkpoints_dir)
        print(f"Cleaned LDS checkpoints from: {checkpoints_dir}")
        
        # Recreate empty directory for future use
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
    else:
        print(f"No LDS checkpoints found at: {checkpoints_dir}")

def get_run_size_info(run_dir: Optional[Path] = None) -> Dict[str, float]:
    """
    Get size information for a run directory.
    
    Args:
        run_dir: Specific run directory to analyze. If None, uses current run.
        
    Returns:
        Dict with size information in GB for different components.
    """
    if run_dir is None:
        run_dir = get_current_run_dir()
    
    def get_dir_size_gb(path: Path) -> float:
        """Get directory size in GB."""
        if not path.exists():
            return 0.0
        total = 0
        for entry in path.rglob('*'):
            if entry.is_file():
                total += entry.stat().st_size
        return total / (1024**3)  # Convert to GB
    
    size_info = {
        'total': get_dir_size_gb(run_dir),
        'magic_checkpoints': get_dir_size_gb(run_dir / "checkpoints_magic"),
        'lds_checkpoints': get_dir_size_gb(run_dir / "checkpoints_lds"),
        'magic_scores': get_dir_size_gb(run_dir / "scores_magic"),
        'lds_losses': get_dir_size_gb(run_dir / "losses_lds"),
        'logs': get_dir_size_gb(run_dir / "logs_magic") + get_dir_size_gb(run_dir / "logs_lds"),
        'plots': get_dir_size_gb(run_dir / "plots_magic") + get_dir_size_gb(run_dir / "plots_lds"),
    }
    
    # Calculate other (batch_dict, indices, etc.)
    size_info['other'] = size_info['total'] - sum(v for k, v in size_info.items() if k != 'total')
    
    return size_info

def save_run_metadata(metadata: Dict[str, Any], run_dir: Optional[Path] = None) -> None:
    """
    Save metadata for a run including configuration snapshot.
    
    Args:
        metadata: Dictionary of metadata to save
        run_dir: Specific run directory. If None, uses current run.
    """
    if run_dir is None:
        run_dir = get_current_run_dir()
    
    metadata_file = run_dir / "run_metadata.json"
    
    # Add timestamp if not present
    if 'timestamp' not in metadata:
        metadata['timestamp'] = datetime.datetime.now().isoformat()
    
    # Handle configuration snapshot
    if 'config_snapshot' not in metadata:
        if 'config' in metadata: # Check for the old key
            metadata['config_snapshot'] = metadata.pop('config') # Move to new key and remove old
            # Do not call get_current_config_dict() if we just moved an existing one
        else:
            # Neither 'config_snapshot' nor 'config' (old key) was present, so create a new snapshot
            metadata['config_snapshot'] = project_config.get_current_config_dict()
    elif 'config' in metadata and 'config_snapshot' in metadata:
        # If by some chance both are present, 'config_snapshot' (the new standard) takes precedence.
        # We can remove the old 'config' key to avoid confusion.
        metadata.pop('config', None) # Remove 'config' if it exists, do nothing if not

    # Save metadata
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

def clean_run_checkpoints(run_id_to_clean: str) -> None:
    """Clean checkpoints from a specific run ID."""
    if not USE_TIMESTAMPED_RUNS:
        print("Cannot clean specific run in flat output mode. Use --clean_magic or --clean_lds for current context.")
        return

    run_dir_to_clean = project_config.OUTPUTS_DIR / RUNS_DIR_NAME / run_id_to_clean
    if not run_dir_to_clean.exists():
        print(f"Run directory {run_dir_to_clean} not found. Nothing to clean.")
        return
    
    print(f"--- Cleaning Checkpoints from Run: {run_id_to_clean} --- ({run_dir_to_clean})")
    clean_magic_checkpoints(run_dir_to_clean)  # Pass the specific run_dir
    clean_lds_checkpoints(run_dir_to_clean)    # Pass the specific run_dir
    print(f"--- Checkpoint Cleaning Complete for Run: {run_id_to_clean} ---")

# --- Output Subdirectories (Standardized) ---
# These are now dynamic based on the current run directory
# These functions will be moved here as well as they depend on get_current_run_dir

def _get_output_dir(subdir: str) -> Path:
    """Get output directory path within current run."""
    return get_current_run_dir() / subdir

# MAGIC Analysis Outputs
def get_magic_checkpoints_dir() -> Path:
    """Get MAGIC checkpoints directory for current run."""
    return _get_output_dir("checkpoints_magic")

def get_magic_scores_dir() -> Path:
    """Get MAGIC scores directory for current run."""
    return _get_output_dir("scores_magic")

def get_magic_plots_dir() -> Path:
    """Get MAGIC plots directory for current run."""
    return _get_output_dir("plots_magic")

def get_magic_logs_dir() -> Path:
    """Get MAGIC logs directory for current run."""
    return _get_output_dir("logs_magic")

def get_batch_dict_file() -> Path:
    """Get batch dict file path for current run."""
    return get_current_run_dir() / "magic_batch_dict.pkl"

# LDS Validator Outputs
def get_lds_checkpoints_dir() -> Path:
    """Get LDS checkpoints directory for current run."""
    return _get_output_dir("checkpoints_lds")

def get_lds_losses_dir() -> Path:
    """Get LDS losses directory for current run."""
    return _get_output_dir("losses_lds")

def get_lds_indices_file() -> Path:
    """Get LDS indices file path for current run."""
    # Directly use the known filename to ensure it's a string for path construction
    # This isolates whether project_config.LDS_INDICES_FILE was the issue
    return get_current_run_dir() / "indices_lds.pkl"

def get_lds_plots_dir() -> Path:
    """Get LDS plots directory for current run."""
    return _get_output_dir("plots_lds")

def get_lds_logs_dir() -> Path:
    """Get LDS logs directory for current run."""
    return _get_output_dir("logs_lds")

# --- File Naming Helper Functions (using Path objects) ---
# These also depend on the get_*_dir functions above

def get_magic_checkpoint_path(model_id: int, step_or_epoch: int) -> Path:
    """
    Get the path for a MAGIC model checkpoint.
    
    Args:
        model_id (int): Model identifier (typically 0 for single model).
        step_or_epoch (int): Training step or epoch number.
        
    Returns:
        Path: Complete path to the checkpoint file.
    """
    return get_magic_checkpoints_dir() / f"sd_{model_id}_{step_or_epoch}.pt"

def get_magic_scores_path(target_idx: int) -> Path:
    """
    Get the path for MAGIC influence scores.
    
    Args:
        target_idx (int): Target validation image index.
        
    Returns:
        Path: Complete path to the scores file.
    """
    return get_magic_scores_dir() / f"magic_scores_val_{target_idx}.pkl"

def get_lds_subset_model_checkpoint_path(model_id: int, step_or_epoch: int) -> Path:
    """
    Get the path for an LDS subset model checkpoint.
    
    Args:
        model_id (int): LDS model identifier.
        step_or_epoch (int): Training step or epoch number.
        
    Returns:
        Path: Complete path to the checkpoint file.
    """
    return get_lds_checkpoints_dir() / f"sd_lds_{model_id}_{step_or_epoch}.pt"

def get_lds_model_val_loss_path(model_id: int) -> Path:
    """
    Get the path for LDS model validation losses.
    
    Args:
        model_id (int): LDS model identifier.
        
    Returns:
        Path: Complete path to the validation loss file.
    """
    return get_lds_losses_dir() / f"loss_lds_{model_id}.pkl"

def get_magic_training_log_path() -> Path:
    """
    Get the path for MAGIC training logs (metrics and hyperparameters).
    
    Returns:
        Path: Complete path to the training log file.
    """
    return get_magic_logs_dir() / "magic_training_log.json"

def get_magic_replay_log_path() -> Path:
    """
    Get the path for MAGIC replay (influence computation) logs.
    
    Returns:
        Path: Complete path to the replay log file.
    """
    return get_magic_logs_dir() / "magic_replay_log.json"

def get_lds_training_log_path(model_id: int) -> Path:
    """
    Get the path for LDS model training logs.
    
    Args:
        model_id (int): LDS model identifier.
        
    Returns:
        Path: Complete path to the training log file.
    """
    return get_lds_logs_dir() / f"lds_training_log_{model_id}.json"

def get_magic_scores_file_for_lds_input() -> Path:
    """Get the MAGIC scores file path that LDS will use."""
    # This depends on LDS_TARGET_VAL_IMAGE_IDX_FOR_CORRELATION from project_config
    return get_magic_scores_path(project_config.LDS_TARGET_VAL_IMAGE_IDX_FOR_CORRELATION)

# Placeholder for get_current_config_dict if it's needed by save_run_metadata
# This function would ideally live in config.py and be imported, or its logic replicated.
# For now, save_run_metadata's config snapshot part is effectively disabled.
# def get_current_config_dict_moved() -> Dict[str, Any]:
#     # Helper to get all relevant config vars
#     # This needs careful implementation to avoid circular deps and get correct scope
#     return {k: v for k, v in project_config.__dict__.items() 
#             if not k.startswith('_') and not callable(v) and not isinstance(v, type(Path))} 