import random
import numpy as np
import torch
import logging
import sys
from typing import Optional, Any, Callable, Union, List, Dict, Iterable
from contextlib import contextmanager
import hashlib
import json
from datetime import datetime
from pathlib import Path

from . import config # Import the config module
from .config import SEED, DEVICE, PAPER_NUM_MEASUREMENT_FUNCTIONS, PAPER_MEASUREMENT_TARGET_INDICES

# Custom Exceptions for better error handling
class DeterministicStateError(Exception):
    """Raised when deterministic state cannot be properly configured."""
    pass

class SeedDerivationError(Exception):
    """Raitten when seed derivation fails."""
    pass

class ComponentCreationError(Exception):
    """Raised when deterministic component creation fails."""
    pass


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    Setup logging configuration for the project.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path to write logs to. If None, logs to console only.
    
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger('influence_analysis')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# =====================================
# MODERN DETERMINISTIC SEED MANAGEMENT
# =====================================

def set_global_deterministic_state(master_seed: int, enable_deterministic: bool = True) -> None:
    """
    Set global deterministic state for all libraries.
    Call this ONCE at the beginning of your program.
    
    Updated for 2025 with latest PyTorch 2.6+ best practices:
    - Enhanced CUBLAS workspace configuration
    - torch.utils.deterministic settings
    - Environment variable management
    
    Based on PyTorch best practices:
    https://pytorch.org/docs/stable/notes/randomness.html
    
    Args:
        master_seed: Master seed for all randomness
        enable_deterministic: Whether to enable strict deterministic mode (slower but reproducible)
    """
    import os
    logger = logging.getLogger('influence_analysis.deterministic')
    logger.info(f"Setting global deterministic state with master seed: {master_seed}")
    
    # Python built-in random
    random.seed(master_seed)
    
    # NumPy random (handle both old and new APIs)
    np.random.seed(master_seed)
    
    # PyTorch CPU
    torch.manual_seed(master_seed)
    
    # PyTorch GPU(s) 
    if torch.cuda.is_available():
        torch.cuda.manual_seed(master_seed)
        torch.cuda.manual_seed_all(master_seed)
        
        # Set CUBLAS workspace config for CUDA 10.2+ (2025 best practice)
        if 'CUBLAS_WORKSPACE_CONFIG' not in os.environ:
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
            logger.debug("Set CUBLAS_WORKSPACE_CONFIG=:4096:8 for deterministic CUDA operations")
    
    if enable_deterministic:
        # Enable deterministic algorithms (2025 update: warn_only=True for better compatibility)
        torch.use_deterministic_algorithms(True, warn_only=True)
        
        # CuDNN deterministic settings
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # 2025 NEW: Enhanced uninitialized memory handling
        if hasattr(torch.utils.deterministic, 'fill_uninitialized_memory'):
            torch.utils.deterministic.fill_uninitialized_memory = True
            logger.debug("Enabled deterministic uninitialized memory filling")
        
        logger.info("Enabled strict deterministic mode (may impact performance)")
    else:
        logger.info("Using standard reproducible mode (faster)")

def derive_component_seed(master_seed: int, purpose: str, instance_id: Optional[Union[str, int]] = None) -> int:
    """
    Derive component-specific seeds from master seed to avoid correlations.
    
    This prevents the issue where all components use the same seed,
    which can create unintended correlations between random operations.
    
    Updated 2025: Enhanced hash function for better distribution
    
    Args:
        master_seed: Master seed
        purpose: Purpose of the component (e.g., 'dataloader', 'model', or a more descriptive string from config)
        instance_id: Optional instance identifier for multiple instances
    
    Returns:
        Derived seed for the component
    """
    # Use hash to derive seeds deterministically but avoid correlations
    seed_string = f"{master_seed}_{purpose}"
    if instance_id is not None:
        seed_string += f"_{instance_id}"
    
    # 2025 IMPROVEMENT: Use SHA256 for better distribution than built-in hash()
    # Built-in hash() can vary between Python sessions, SHA256 is deterministic
    hash_object = hashlib.sha256(seed_string.encode())
    derived_seed = int(hash_object.hexdigest()[:8], 16) % (2**31)  # Keep it positive 32-bit int
    
    logger = logging.getLogger('influence_analysis.deterministic')
    logger.debug(f"Derived seed for purpose '{purpose}' (instance: '{instance_id}') = {derived_seed}")
    
    return derived_seed

def seed_worker(worker_id: int) -> None:
    """
    Worker initialization function for DataLoader.
    
    Based on PyTorch 2.6+ best practices for deterministic DataLoader.
    This ensures each worker has a deterministic but unique seed.
    
    Updated 2025: Enhanced worker seed derivation
    
    Args:
        worker_id: Worker ID assigned by DataLoader
    """
    # Get the base seed from the main process (PyTorch 2.6+ approach)
    worker_seed = torch.initial_seed() % 2**32
    
    # Set seeds for this worker
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
    # 2025 NEW: Also set torch seed for worker (in case worker uses torch random ops)
    torch.manual_seed(worker_seed)
    
    # Optional: Log for debugging (usually too verbose)
    # logger = logging.getLogger('influence_analysis.worker')
    # logger.debug(f"Worker {worker_id} initialized with seed {worker_seed}")

@contextmanager
def deterministic_context(component_seed: int, component_name: str = "operation"):
    """
    Lightweight context manager for deterministic operations.
    
    Unlike the old approach, this only sets PyTorch seeds and doesn't 
    interfere with global Python/NumPy state.
    
    Updated 2025: Enhanced state preservation and error handling
    
    Args:
        component_seed: Seed for this specific component
        component_name: Name of component (for logging)
    """
    logger = logging.getLogger('influence_analysis.deterministic')
    logger.debug(f"Entering deterministic context for {component_name} with seed {component_seed}")
    
    # Only set PyTorch seeds (most critical for model operations)
    old_rng_state = torch.get_rng_state()
    old_cuda_rng_state = None
    
    # 2025 IMPROVEMENT: Better CUDA state handling
    try:
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            old_cuda_rng_state = torch.cuda.get_rng_state_all()
    except RuntimeError as e:
        logger.warning(f"Could not save CUDA RNG state: {e}")
    
    torch.manual_seed(component_seed)
    if torch.cuda.is_available():
        try:
            torch.cuda.manual_seed_all(component_seed)
        except RuntimeError as e:
            logger.warning(f"Could not set CUDA seed: {e}")
    
    try:
        yield
    finally:
        # Restore previous state instead of leaving seeds modified
        try:
            torch.set_rng_state(old_rng_state)
            if torch.cuda.is_available() and old_cuda_rng_state is not None:
                torch.cuda.set_rng_state_all(old_cuda_rng_state)
        except RuntimeError as e:
            logger.warning(f"Could not restore RNG state: {e}")
        
        logger.debug(f"Exited deterministic context for {component_name}")

# =====================================
# ADVANCED DETERMINISTIC SAMPLER
# =====================================

class DeterministicSampler(torch.utils.data.Sampler):
    """
    Deterministic sampler that produces identical sequences across instances.
    
    This solves the PyTorch DataLoader consistency issue by managing
    the sampling process explicitly rather than relying on internal shuffling.
    """
    
    def __init__(self, dataset_size: int, shuffle: bool = True, seed: int = 42, epoch: int = 0):
        self.dataset_size = dataset_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = epoch
        
    def __iter__(self):
        if self.shuffle:
            # Create deterministic permutation using component-specific seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)  # Include epoch for multi-epoch consistency
            indices = torch.randperm(self.dataset_size, generator=g).tolist()
        else:
            indices = list(range(self.dataset_size))
        return iter(indices)
    
    def __len__(self):
        return self.dataset_size
    
    def set_epoch(self, epoch: int):
        """Set epoch for multi-epoch deterministic shuffling"""
        self.epoch = epoch

def create_deterministic_dataloader(
    master_seed: int, 
    creator_func: Callable, 
    instance_id: Optional[Union[str, int]] = None, 
    **kwargs
) -> Any:
    """
    Create a deterministic DataLoader with perfect consistency guarantee.
        
    Args:
        master_seed: Master seed
        creator_func: Function that creates the dataloader
        instance_id: Optional instance identifier. Used to differentiate seeds or ensure identity.
        **kwargs: Arguments to pass to creator_func (e.g., batch_size, split, shuffle)
    
    Returns:
        Created dataloader with guaranteed deterministic behavior
    """
    logger = logging.getLogger('influence_analysis.deterministic')
    log_instance_name = f"instance: {instance_id}" if instance_id else "default instance"
    logger.debug(f"Creating deterministic dataloader ({log_instance_name})")
    
    # Use fixed "dataloader" string for type part of seed derivation.
    component_seed = derive_component_seed(master_seed, "dataloader", instance_id)
    
    # Create base dataloader to get dataset and parameters
    # The initial call to creator_func also needs to be in a deterministic context if it has any random ops
    context_name_initial = f"dataloader_initial_creation ({log_instance_name})"
    with deterministic_context(component_seed, context_name_initial):
        base_dataloader = creator_func(**kwargs)
    
    # Extract parameters for deterministic recreation
    batch_size = getattr(base_dataloader, 'batch_size', 1)
    num_workers = getattr(base_dataloader, 'num_workers', 0)
    pin_memory = getattr(base_dataloader, 'pin_memory', False)
    drop_last = getattr(base_dataloader, 'drop_last', False)
    shuffle = kwargs.get('shuffle', False) # Get shuffle from original kwargs for sampler
    
    sampler = DeterministicSampler(
        dataset_size=len(base_dataloader.dataset),
        shuffle=shuffle,
        seed=component_seed, # Sampler uses the main component_seed
        epoch=0
    )
    
    generator = torch.Generator()
    generator.manual_seed(component_seed) # Generator also uses the main component_seed
    
    deterministic_dataloader = torch.utils.data.DataLoader(
        dataset=base_dataloader.dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=base_dataloader.collate_fn,
        pin_memory=pin_memory,
        drop_last=drop_last,
        worker_init_fn=seed_worker, # seed_worker derives its own seed from torch.initial_seed()
        generator=generator,
        persistent_workers=getattr(base_dataloader, 'persistent_workers', False)
    )
    
    deterministic_dataloader._deterministic_sampler = sampler
    
    logger.info(f"Created deterministic dataloader ({log_instance_name}) with component seed {component_seed}")
    return deterministic_dataloader

def create_deterministic_model(
    master_seed: int, 
    creator_func: Callable, 
    instance_id: Optional[Union[str, int]] = None, 
    **kwargs
) -> Any:
    """
    Create a deterministic model instance.
    
    Ensures consistent initialization by managing seeds and context.
    This function is crucial for reproducible research.
    
    Updated 2025: Streamlined context management
    
    Args:
        master_seed: Master seed
        creator_func: Function that creates the model (e.g., a constructor)
        instance_id: Optional instance identifier for unique seed derivation
        **kwargs: Additional arguments passed to the creator_func
    
    Returns:
        Created model instance
    """
    component_seed = derive_component_seed(master_seed, "model", instance_id)
    logger = logging.getLogger('influence_analysis.deterministic')
    logger.info(f"Creating model '{creator_func.__name__}' with seed {component_seed} (instance: '{instance_id}')")
    
    with deterministic_context(component_seed, f"model_{creator_func.__name__}_{instance_id}"):
        try:
            # Pass NUM_CLASSES from config to the model creator function
            # Architecture-specific parameters (width_multiplier, etc.) are now defaulted
            # in the model definitions themselves, sourcing from config.py.
            # We only pass them here if explicitly provided in **kwargs for overrides.
            
            # Ensure num_classes is always passed, defaulting to config.NUM_CLASSES
            # if not in kwargs. Other model-specific args from config are now handled by model defs.
            model_kwargs = {'num_classes': config.NUM_CLASSES, **kwargs}
            
            # Old logic that explicitly passed config values:
            # if creator_func.__name__ == 'ResNet9TableArch':
            #     model = creator_func(
            #         num_classes=config.NUM_CLASSES,
            #         width_multiplier=config.RESNET9_WIDTH_MULTIPLIER,
            #         pooling_epsilon=config.RESNET9_POOLING_EPSILON,
            #         final_layer_scale=config.RESNET9_FINAL_LAYER_SCALE,
            #         **kwargs
            #     )
            # elif creator_func.__name__ == 'make_airbench94_adapted':
            #     model = creator_func(
            #         width_multiplier=config.RESNET9_WIDTH_MULTIPLIER, # Reusing RESNET9_WIDTH_MULTIPLIER
            #         pooling_epsilon=config.RESNET9_POOLING_EPSILON,   # Reusing RESNET9_POOLING_EPSILON
            #         final_layer_scale=config.RESNET9_FINAL_LAYER_SCALE, # Reusing RESNET9_FINAL_LAYER_SCALE
            #         num_classes=config.NUM_CLASSES, # Pass num_classes
            #         **kwargs
            #     )
            # else: # Existing behavior for other models like construct_resnet9_paper or construct_rn9
            #     model = creator_func(num_classes=config.NUM_CLASSES, **kwargs) 

            model = creator_func(**model_kwargs)
            logger.info(f"Model '{creator_func.__name__}' created successfully with args: {model_kwargs}.")
            return model
        except Exception as e:
            logger.error(f"Error creating model '{creator_func.__name__}': {e}")
            raise ComponentCreationError(f"Failed to create model '{creator_func.__name__}': {e}") from e

def create_deterministic_optimizer(
    master_seed: int, 
    optimizer_class: type, 
    model_params: Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]], # Allow pre-grouped params
    instance_id: Optional[Union[str, int]] = None, 
    **kwargs
) -> Any:
    """
    Create an optimizer with deterministic initialization.
    
    Args:
        master_seed: Master seed
        optimizer_class: Optimizer class (e.g., torch.optim.SGD)
        model_params: Model parameters to optimize, or an iterable of dicts for param groups.
        instance_id: Optional instance identifier.
        **kwargs: Arguments to pass to optimizer constructor
    
    Returns:
        Created optimizer
    """
    logger = logging.getLogger('influence_analysis.deterministic')
    log_instance_name = f"instance: {instance_id}" if instance_id else "default instance"
    logger.debug(f"Creating deterministic optimizer ({log_instance_name})")

    # Use fixed "optimizer" string for type part of seed derivation.
    component_seed = derive_component_seed(master_seed, "optimizer", instance_id)
    
    context_name = f"optimizer_creation ({log_instance_name})"
    with deterministic_context(component_seed, context_name):
        optimizer = optimizer_class(model_params, **kwargs)
    
    logger.info(f"Created deterministic optimizer ({log_instance_name}) with component seed {component_seed}")
    return optimizer

def create_deterministic_scheduler(
    master_seed: int, 
    optimizer: torch.optim.Optimizer, 
    schedule_type: Optional[str],
    total_steps: int, 
    instance_id: Optional[Union[str, int]] = None,
    three_phase: bool = False, # Added for OneCycleLR
    **scheduler_params
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    Create a learning rate scheduler with deterministic initialization.
    
    Args:
        master_seed: Master seed
        optimizer: PyTorch optimizer
        schedule_type: Type of scheduler ('StepLR', 'CosineAnnealingLR', 'OneCycleLR', or None)
        total_steps: Total number of training steps
        instance_id: Optional instance identifier.
        three_phase: Boolean indicating if the scheduler is three_phase (for OneCycleLR)
        **scheduler_params: Additional parameters for specific schedulers
    
    Returns:
        Configured scheduler or None if no scheduling
    """
    if schedule_type is None:
        return None
        
    logger = logging.getLogger('influence_analysis.deterministic')
    log_instance_name = f"instance: {instance_id}" if instance_id else "default instance"
    logger.debug(f"Creating deterministic scheduler ({log_instance_name})")
    
    # Use fixed "scheduler" string for type part of seed derivation.
    component_seed = derive_component_seed(master_seed, "scheduler", instance_id)
    
    context_name = f"scheduler_creation ({log_instance_name})"
    with deterministic_context(component_seed, context_name):
        if schedule_type == 'OneCycleLR':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=scheduler_params.get('max_lr', 0.1),
                total_steps=total_steps,
                pct_start=scheduler_params.get('pct_start', 0.3),
                anneal_strategy=scheduler_params.get('anneal_strategy', 'cos'),
                div_factor=scheduler_params.get('div_factor', 25.0),
                final_div_factor=scheduler_params.get('final_div_factor', 10000.0),
                three_phase=scheduler_params.get('three_phase', False) # Ensure three_phase is correctly fetched
            )
        elif schedule_type == 'CosineAnnealingLR':
            t_max = scheduler_params.get('t_max')
            if t_max is None:
                t_max = total_steps
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=t_max
            )
        elif schedule_type == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=scheduler_params.get('step_size', 10),
                gamma=scheduler_params.get('gamma', 0.1)
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {schedule_type}")
    
    logger.info(f"Created deterministic scheduler ({log_instance_name}) with component seed {component_seed}")
    return scheduler

def create_scheduler(optimizer: torch.optim.Optimizer, schedule_type: Optional[str], 
                    total_steps: int, **scheduler_params) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    Factory function to create learning rate schedulers with consistent configuration.
    
    Args:
        optimizer: PyTorch optimizer
        schedule_type: Type of scheduler ('StepLR', 'CosineAnnealingLR', 'OneCycleLR', or None)
        total_steps: Total number of training steps (used for schedulers that need it)
        **scheduler_params: Additional parameters for specific schedulers
    
    Returns:
        Configured scheduler or None if no scheduling
    """
    if schedule_type is None:
        return None
        
    if schedule_type == 'StepLR':
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_params.get('step_size', 10),
            gamma=scheduler_params.get('gamma', 0.1)
        )
    elif schedule_type == 'CosineAnnealingLR':
        t_max = scheduler_params.get('t_max')
        if t_max is None:
            t_max = total_steps
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=t_max
        )
    elif schedule_type == 'OneCycleLR':
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=scheduler_params.get('max_lr', 0.1),
            total_steps=total_steps,
            pct_start=scheduler_params.get('pct_start', 0.3),
            anneal_strategy=scheduler_params.get('anneal_strategy', 'cos'),
            div_factor=scheduler_params.get('div_factor', 25.0),
            final_div_factor=scheduler_params.get('final_div_factor', 10000.0)
        )
    else:
        raise ValueError(f"Unsupported scheduler type: {schedule_type}")

def update_dataloader_epoch(dataloader: torch.utils.data.DataLoader, epoch: int) -> None:
    """
    Update the epoch for deterministic dataloader to ensure proper multi-epoch shuffling.
    
    Args:
        dataloader: DataLoader with deterministic sampler
        epoch: Current epoch number
    """
    if hasattr(dataloader, '_deterministic_sampler'):
        dataloader._deterministic_sampler.set_epoch(epoch)

# =====================================
# CONFIGURATION LOGGING UTILITIES
# =====================================

def get_run_config_as_dict(
    component_type: str, # "MAGIC" or "LDS"
    device: torch.device,
    model_creator_func: Optional[Callable],
    model: torch.nn.Module, # Actual instantiated model
    optimizer: torch.optim.Optimizer, # Actual instantiated optimizer
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler], # Actual instantiated scheduler
    train_loader: torch.utils.data.DataLoader, # Actual instantiated train_loader
    val_loader: Optional[torch.utils.data.DataLoader], # Actual instantiated val_loader (optional for LDS if not used pre-config)
    total_epochs_for_run: int,
    steps_per_epoch: int,
    effective_lr_base: float, # Base LR for the run
    # MAGIC specific (optional)
    magic_target_val_image_idx: Optional[int] = None,
    magic_num_influential_images_to_show: Optional[int] = None,
    magic_is_memory_efficient_replay: Optional[bool] = None,
    # LDS specific (optional)
    lds_model_id: Optional[int] = None,
    lds_subset_size: Optional[int] = None,
    lds_subset_indices_list: Optional[list] = None,
    lds_subset_fraction: Optional[float] = None
) -> Dict[str, Any]:
    """
    Assembles a comprehensive configuration dictionary for a training run,
    including both configured and actual instantiated states.
    Used by both MAGIC and LDS to ensure consistent logging of configurations.
    """
    config_data = {
        # Core system configuration
        "component_type": component_type,
        "device": str(device),
        "seed": config.SEED, # Global seed
        "cuda_available": torch.cuda.is_available(),
        
        # Model architecture configuration (configured)
        "model_creator_function_config": model_creator_func.__name__ if model_creator_func else None,
        "num_classes_config": config.NUM_CLASSES,
        
        # Model details (actual instantiated)
        "actual_model_class": model.__class__.__name__,
        "actual_model_total_params": sum(p.numel() for p in model.parameters()),
        "actual_model_trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "actual_model_memory_format": "channels_last" if str(device) == 'cuda' and next(model.parameters()).is_cuda and next(model.parameters()).is_contiguous(memory_format=torch.channels_last) else "contiguous",

        # Training hyperparameters (effective for this run)
        "run_total_epochs": total_epochs_for_run,
        "run_steps_per_epoch": steps_per_epoch,
        "run_total_training_steps": total_epochs_for_run * steps_per_epoch,
        "run_effective_lr_base": effective_lr_base,
        "run_train_batch_size_config": config.MODEL_TRAIN_BATCH_SIZE, 
            
        # Optimizer details (configured from global config)
        "optimizer_type_config": config.MODEL_TRAIN_OPTIMIZER if component_type == "MAGIC" else "SGD", # LDS effectively uses SGD
        "optimizer_momentum_config": config.MODEL_TRAIN_MOMENTUM,
        "optimizer_weight_decay_config": config.MODEL_TRAIN_WEIGHT_DECAY,
        "optimizer_nesterov_config": config.MODEL_TRAIN_NESTEROV,

        # Optimizer details (actual instantiated)
        "actual_optimizer_class": optimizer.__class__.__name__,
        "actual_optimizer_initial_lr": optimizer.param_groups[0]['lr'] if optimizer.param_groups else None,
        "actual_optimizer_param_groups_count": len(optimizer.param_groups),
        "actual_optimizer_state_dict_keys": list(optimizer.state_dict().keys()),
        "actual_optimizer_defaults": dict(optimizer.defaults) if hasattr(optimizer, 'defaults') else {},
        
        # Explicit Optimizer Parameter Group Details
        "actual_optimizer_param_group_details": [], # Initialize
        
        # Scheduler details (actual instantiated) - handled by get_scheduler_config_as_dict
        # "actual_scheduler_class": scheduler.__class__.__name__ if scheduler else None,
        # "actual_scheduler_state_dict_keys": list(scheduler.state_dict().keys()) if scheduler else None,

        # Data configuration (global and actual)
        "num_train_samples_global_config": config.NUM_TRAIN_SAMPLES,
        "cifar_root_global_config": str(config.CIFAR_ROOT),
        "dataloader_num_workers_global_config": config.DATALOADER_NUM_WORKERS,
        "actual_train_loader_batch_size": train_loader.batch_size,
        "actual_train_loader_num_workers": train_loader.num_workers,
        "actual_train_loader_dataset_size": len(train_loader.dataset) if train_loader and hasattr(train_loader, 'dataset') else None,
        
        "actual_val_loader_batch_size": val_loader.batch_size if val_loader else None,
        "actual_val_loader_dataset_size": len(val_loader.dataset) if val_loader and hasattr(val_loader, 'dataset') else None,
        
        # Shared instance IDs for reproducibility
        "shared_model_instance_id": config.SHARED_MODEL_INSTANCE_ID,
        "shared_dataloader_instance_id": config.SHARED_DATALOADER_INSTANCE_ID, # For train_loader
        "shared_optimizer_instance_id": config.SHARED_OPTIMIZER_INSTANCE_ID,
        
        "epsilon_for_weighted_loss_global": config.EPSILON_FOR_WEIGHTED_LOSS,
        "perform_inline_validations_global": config.PERFORM_INLINE_VALIDATIONS,
    }

    if hasattr(optimizer, 'param_groups'):
        param_group_details_list = []
        for i, group in enumerate(optimizer.param_groups):
            group_detail = {
                "group_index": i,
                "lr": group.get('lr'),
                "initial_lr": group.get('initial_lr', group.get('lr')), # For OneCycleLR, initial_lr is important
                "weight_decay": group.get('weight_decay', optimizer.defaults.get('weight_decay', 0)),
                "momentum": group.get('momentum', optimizer.defaults.get('momentum', 0)),
                "nesterov": group.get('nesterov', optimizer.defaults.get('nesterov', False)),
                "num_params_in_group": sum(p.numel() for p in group['params'])
            }
            param_group_details_list.append(group_detail)
        config_data["actual_optimizer_param_group_details"] = param_group_details_list

    if model_creator_func and model_creator_func.__name__ == 'construct_resnet9_paper':
        config_data.update({
            "resnet9_width_multiplier_config": config.RESNET9_WIDTH_MULTIPLIER,
            "resnet9_bias_scale_config": config.RESNET9_BIAS_SCALE,
            "resnet9_final_layer_scale_config": config.RESNET9_FINAL_LAYER_SCALE,
            "resnet9_pooling_epsilon_config": config.RESNET9_POOLING_EPSILON,
        })

    if component_type == "MAGIC":
        config_data.update({
            "magic_target_val_image_idx_config": magic_target_val_image_idx,
            "magic_num_influential_images_to_show_config": magic_num_influential_images_to_show,
            "magic_memory_efficient_replay_config": magic_is_memory_efficient_replay,
        })
    elif component_type == "LDS":
        config_data.update({
            "lds_model_id": lds_model_id,
            "lds_subset_size": lds_subset_size,
            "lds_subset_indices_list_preview": lds_subset_indices_list[:10] if lds_subset_indices_list else None,
            "lds_subset_indices_list_len": len(lds_subset_indices_list) if lds_subset_indices_list else 0,
            "lds_subset_fraction": lds_subset_fraction,
        })

    config_data['scheduler_config'] = get_scheduler_config_as_dict(
        scheduler,
        optimizer, 
        total_epochs_for_run 
    )
    
    return config_data

def get_scheduler_config_as_dict(scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                                 optimizer: torch.optim.Optimizer,
                                 total_epochs_for_current_run: int) -> Dict[str, Any]:
    """
    Inspects a PyTorch scheduler and optimizer to create a detailed configuration dictionary for logging.

    Args:
        scheduler: The instantiated learning rate scheduler object (can be None).
        optimizer: The optimizer associated with the scheduler.
        total_epochs_for_current_run: The total number of epochs this scheduler is intended for.

    Returns:
        A dictionary detailing the scheduler's configuration.
    """
    scheduler_config_log = {}
    if scheduler is None:
        scheduler_config_log['type'] = None
        scheduler_config_log['details'] = "Constant LR"
        scheduler_config_log['constant_lr_used'] = optimizer.param_groups[0]['lr']
    elif isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
        scheduler_config_log['type'] = 'OneCycleLR'
        # Log all max_lrs and initial_lrs if they are lists (due to param groups)
        max_lrs_effective = [group['max_lr'] for group in scheduler.optimizer.param_groups]
        initial_lrs_calculated = [pg.get('initial_lr', pg['lr']) for pg in scheduler.optimizer.param_groups]

        scheduler_config_log['details'] = {
            'max_lrs_effective': max_lrs_effective,
            'total_steps': scheduler.total_steps,
            'pct_start': config.ONECYCLE_PCT_START,
            'anneal_strategy': config.ONECYCLE_ANNEAL_STRATEGY,
            'div_factor': config.ONECYCLE_DIV_FACTOR,
            'final_div_factor': config.ONECYCLE_FINAL_DIV_FACTOR,
            'initial_lrs_calculated': initial_lrs_calculated
        }
        scheduler_config_log['shared_instance_id'] = config.SHARED_SCHEDULER_INSTANCE_ID
    elif isinstance(scheduler, torch.optim.lr_scheduler.SequentialLR):
        scheduler_config_log['type'] = 'SequentialLR'
        warmup_sched_obj = scheduler._schedulers[0]
        main_sched_obj = scheduler._schedulers[1]
        milestone_val = scheduler._milestones[0]
        warmup_details_log = {}
        if isinstance(warmup_sched_obj, torch.optim.lr_scheduler.LinearLR):
            warmup_details_log = {
                'type': 'LinearLR',
                'start_factor': warmup_sched_obj.start_factor,
                'end_factor': warmup_sched_obj.end_factor,
                'total_iters': warmup_sched_obj.total_iters,
                'duration_epochs_config': config.WARMUP_EPOCHS # Based on global config
            }
        main_details_log = {'type': main_sched_obj.__class__.__name__}
        if isinstance(main_sched_obj, torch.optim.lr_scheduler.StepLR):
            main_details_log.update({
                'step_size_epochs_effective': main_sched_obj.step_size,
                'gamma': main_sched_obj.gamma
            })
        elif isinstance(main_sched_obj, torch.optim.lr_scheduler.CosineAnnealingLR):
            main_details_log.update({
                'T_max_steps_effective': main_sched_obj.T_max
            })
        scheduler_config_log['details'] = {
            'warmup_scheduler': warmup_details_log,
            'main_scheduler': main_details_log,
            'milestone_step': milestone_val
        }
        scheduler_config_log['main_scheduler_shared_instance_id_suffix'] = "_main"
    elif isinstance(scheduler, torch.optim.lr_scheduler.LinearLR): # Full warmup case
        scheduler_config_log['type'] = 'LinearLR_WarmupOnly'
        scheduler_config_log['details'] = {
            'start_factor': scheduler.start_factor,
            'end_factor': scheduler.end_factor,
            'total_iters': scheduler.total_iters,
            'duration_epochs_effective': total_epochs_for_current_run
        }
    elif isinstance(scheduler, torch.optim.lr_scheduler.StepLR):
        scheduler_config_log['type'] = 'StepLR'
        scheduler_config_log['details'] = {
            'step_size_epochs_effective': scheduler.step_size,
            'gamma': scheduler.gamma
        }
        scheduler_config_log['shared_instance_id'] = config.SHARED_SCHEDULER_INSTANCE_ID
    elif isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
        scheduler_config_log['type'] = 'CosineAnnealingLR'
        scheduler_config_log['details'] = {
            'T_max_steps_effective': scheduler.T_max
        }
        scheduler_config_log['shared_instance_id'] = config.SHARED_SCHEDULER_INSTANCE_ID
    else: # Fallback for any other scheduler type not explicitly handled
        scheduler_config_log['type'] = scheduler.__class__.__name__ if scheduler else "UnknownType"
        scheduler_config_log['details'] = "Generic scheduler type; inspect object state directly if needed."
        # Attempt to log common attributes if they exist, otherwise omit
        if hasattr(scheduler, 'optimizer'): # Basic check
            scheduler_config_log['base_lrs'] = getattr(scheduler, 'base_lrs', [pg['lr'] for pg in optimizer.param_groups])
            scheduler_config_log['last_epoch'] = getattr(scheduler, 'last_epoch', -1)

    return scheduler_config_log

def log_environment_info(save_metrics_func: Callable, 
                           component_logger: logging.Logger, 
                           model_id_for_metrics: Optional[int] = None) -> None:
    """
    Captures and logs environment information using the provided save_metrics_func.

    Args:
        save_metrics_func: The function to call to save the metrics 
                             (e.g., MagicAnalyzer._save_training_metrics or save_lds_training_metrics).
        component_logger: Logger instance of the calling component.
        model_id_for_metrics: Optional model_id if the save_metrics_func requires it (for LDS).
    """
    try:
        env_info = config.validate_environment() # Assuming this returns a dict
        if model_id_for_metrics is not None and callable(save_metrics_func) and save_metrics_func.__name__ == 'save_lds_training_metrics':
            save_metrics_func(model_id_for_metrics, env_info, "environment")
        elif callable(save_metrics_func):
             save_metrics_func(env_info, "environment")
        else:
            component_logger.error("Provided save_metrics_func is not callable for environment logging.")
            return
        component_logger.debug("Environment information logged.")
    except Exception as e:
        component_logger.warning(f"Failed to capture or log complete environment info: {e}")
        # Save basic environment info as fallback
        basic_env = {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
        try:
            if model_id_for_metrics is not None and callable(save_metrics_func) and save_metrics_func.__name__ == 'save_lds_training_metrics':
                save_metrics_func(model_id_for_metrics, basic_env, "environment_fallback")
            elif callable(save_metrics_func):
                save_metrics_func(basic_env, "environment_fallback")
            component_logger.debug("Basic fallback environment information logged.")
        except Exception as fallback_e:
            component_logger.error(f"Failed to save even basic environment info: {fallback_e}")

# =====================================
# EFFECTIVE SCHEDULER CREATION
# =====================================

def create_effective_scheduler(
    optimizer: torch.optim.Optimizer,
    master_seed: int,
    shared_scheduler_instance_id: str,
    total_epochs_for_run: int,
    steps_per_epoch_for_run: int,
    effective_lr_for_run: Union[float, List[float]], # Allow list for OneCycleLR with groups
    component_logger: logging.Logger,
    component_name: str = "TrainingComponent" # e.g., "MAGIC" or "LDS Model {model_id}"
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    Centralized logic to create the effective learning rate scheduler based on global config 
    and run-specific parameters.
    Logs the decision process using the provided component_logger.

    Args:
        optimizer: The PyTorch optimizer.
        master_seed: The master seed for reproducibility.
        shared_scheduler_instance_id: Base instance ID for deterministic scheduler creation.
        total_epochs_for_run: Total epochs for the current training run.
        steps_per_epoch_for_run: Steps per epoch for the current training run.
        effective_lr_for_run: The base/target learning rate for this run.
        component_logger: The logger instance of the calling component (MagicAnalyzer/LDSValidator).
        component_name: A string name for the component for logging clarity.

    Returns:
        The instantiated scheduler object, or None.
    """
    scheduler = None
    warmup_start_lr_fraction = 0.1
    total_steps_for_run = total_epochs_for_run * steps_per_epoch_for_run

    if config.LR_SCHEDULE_TYPE == 'OneCycleLR':
        component_logger.info(f"{component_name}: Using OneCycleLR scheduler (handles its own warmup).")
        scheduler = create_deterministic_scheduler(
            master_seed=master_seed,
            optimizer=optimizer,
            schedule_type='OneCycleLR',
            total_steps=total_steps_for_run,
            instance_id=shared_scheduler_instance_id,
            max_lr=effective_lr_for_run,
            pct_start=config.ONECYCLE_PCT_START,
            anneal_strategy=config.ONECYCLE_ANNEAL_STRATEGY,
            div_factor=config.ONECYCLE_DIV_FACTOR,
            final_div_factor=config.ONECYCLE_FINAL_DIV_FACTOR,
            three_phase=False # Explicitly False for the main OneCycleLR usage
        )
    elif config.LR_SCHEDULE_TYPE and config.WARMUP_EPOCHS > 0 and total_epochs_for_run > config.WARMUP_EPOCHS:
        component_logger.info(f"{component_name}: Using SequentialLR with {config.WARMUP_EPOCHS} warmup epochs, followed by {config.LR_SCHEDULE_TYPE}.")
        warmup_iters = config.WARMUP_EPOCHS * steps_per_epoch_for_run
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_start_lr_fraction, end_factor=1.0, total_iters=warmup_iters
        )
        main_scheduler_epochs = total_epochs_for_run - config.WARMUP_EPOCHS
        main_scheduler_total_steps = main_scheduler_epochs * steps_per_epoch_for_run
        
        specific_main_scheduler_params = {
            'step_size': config.STEPLR_STEP_SIZE, 'gamma': config.STEPLR_GAMMA,
            't_max': main_scheduler_total_steps if config.LR_SCHEDULE_TYPE == 'CosineAnnealingLR' else None,
            'max_lr': effective_lr_for_run, 'pct_start': config.ONECYCLE_PCT_START,
            'anneal_strategy': config.ONECYCLE_ANNEAL_STRATEGY, 'div_factor': config.ONECYCLE_DIV_FACTOR,
            'final_div_factor': config.ONECYCLE_FINAL_DIV_FACTOR
        }
        # Clean up params not relevant to the main scheduler type chosen
        if config.LR_SCHEDULE_TYPE != 'CosineAnnealingLR': 
            specific_main_scheduler_params.pop('t_max', None)
        if config.LR_SCHEDULE_TYPE != 'StepLR':
            specific_main_scheduler_params.pop('step_size', None)
            specific_main_scheduler_params.pop('gamma', None)
        # OneCycleLR params are generally not used as a main scheduler in SequentialLR context here
        # but kept in dict for safety, create_deterministic_scheduler should ignore them if not OneCycleLR.

        main_scheduler = create_deterministic_scheduler(
            master_seed=master_seed, optimizer=optimizer, schedule_type=config.LR_SCHEDULE_TYPE,
            total_steps=main_scheduler_total_steps, 
            instance_id=f"{shared_scheduler_instance_id}_main",
            # Pass three_phase if main_scheduler could be OneCycleLR, default to False
            three_phase=specific_main_scheduler_params.get('three_phase', False),
            **specific_main_scheduler_params
        )
        if main_scheduler:
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_iters]
            )
        else:
            component_logger.warning(f"{component_name}: Main scheduler part of SequentialLR was None. Using only warmup scheduler.")
            scheduler = warmup_scheduler
    elif config.LR_SCHEDULE_TYPE and config.WARMUP_EPOCHS > 0: # Warmup requested, but total_epochs_for_run <= warmup_epochs
        component_logger.info(f"{component_name}: Warmup epochs ({config.WARMUP_EPOCHS}) >= total epochs ({total_epochs_for_run}). Using LinearLR for full duration.")
        effective_warmup_iters = total_steps_for_run
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_start_lr_fraction, end_factor=1.0, total_iters=effective_warmup_iters
        )
    elif config.LR_SCHEDULE_TYPE: # A schedule type is set, but no warmup (WARMUP_EPOCHS == 0)
        component_logger.info(f"{component_name}: Using {config.LR_SCHEDULE_TYPE} scheduler directly (no warmup).")
        scheduler = create_deterministic_scheduler(
            master_seed=master_seed, optimizer=optimizer, schedule_type=config.LR_SCHEDULE_TYPE,
            total_steps=total_steps_for_run, 
            instance_id=shared_scheduler_instance_id,
            # Pass all potential params; create_deterministic_scheduler will pick the relevant ones
            step_size=config.STEPLR_STEP_SIZE, gamma=config.STEPLR_GAMMA,
            t_max=config.COSINE_T_MAX, # Usually overridden by total_steps for Cosine by create_scheduler
            max_lr=effective_lr_for_run, pct_start=config.ONECYCLE_PCT_START,
            anneal_strategy=config.ONECYCLE_ANNEAL_STRATEGY,
            div_factor=config.ONECYCLE_DIV_FACTOR, final_div_factor=config.ONECYCLE_FINAL_DIV_FACTOR,
            three_phase=False # Explicitly False for this direct scheduler usage
        )
    else: # No LR_SCHEDULE_TYPE defined
        component_logger.info(f"{component_name}: No learning rate scheduler configured (LR_SCHEDULE_TYPE is None). Constant LR will be used.")
        
    return scheduler

# =====================================
# PAPER SPECIFICATION: 50 MEASUREMENT FUNCTIONS
# =====================================

def get_measurement_function_targets() -> List[int]:
    """
    Get the list of target indices for the 50 measurement functions φi.
    
    According to the paper: "we consider 50 measurement functions φi corresponding to 
    loss on 50 different CIFAR-10 test samples"
    
    Returns:
        List[int]: List of 50 target indices [0, 1, 2, ..., 49]
    """
    return PAPER_MEASUREMENT_TARGET_INDICES.copy()


def evaluate_measurement_functions(
    model: torch.nn.Module, 
    test_dataset: torch.utils.data.Dataset,
    target_indices: Optional[List[int]] = None,
    device: torch.device = DEVICE
) -> Dict[int, float]:
    """
    Evaluate the 50 measurement functions φi on a trained model.
    
    Each measurement function φi computes the loss on a specific CIFAR-10 test sample.
    This implements the measurement functions mentioned in the paper.
    
    Args:
        model: Trained PyTorch model to evaluate
        test_dataset: CIFAR-10 test dataset
        target_indices: List of target indices to evaluate. If None, uses all 50 from paper.
        device: Computing device (CPU/GPU)
        
    Returns:
        Dict[int, float]: Dictionary mapping target_idx -> loss value
        
    Example:
        >>> model = construct_resnet9_jor24a_exact()
        >>> test_ds = CustomDataset(root=config.CIFAR_ROOT, train=False)
        >>> losses = evaluate_measurement_functions(model, test_ds)
        >>> print(f"Loss on target 0: {losses[0]:.4f}")
    """
    if target_indices is None:
        target_indices = get_measurement_function_targets()
    
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    measurement_results = {}
    
    with torch.no_grad():
        for target_idx in target_indices:
            if target_idx >= len(test_dataset):
                raise ValueError(f"Target index {target_idx} out of bounds for test dataset (size: {len(test_dataset)})")
            
            # Get the specific test sample
            image, label, _ = test_dataset[target_idx]
            image = image.unsqueeze(0).to(device)  # Add batch dimension
            label = torch.tensor([label]).to(device)
            
            # Compute loss for this specific sample (measurement function φi)
            output = model(image)
            loss = criterion(output, label)
            measurement_results[target_idx] = loss.item()
    
    return measurement_results


def create_measurement_function_config() -> Dict[str, Any]:
    """
    Create a configuration dictionary for the 50 measurement functions.
    
    Returns:
        Dict[str, Any]: Configuration for measurement functions including all paper specifications
    """
    return {
        'num_measurement_functions': PAPER_NUM_MEASUREMENT_FUNCTIONS,
        'target_indices': get_measurement_function_targets(),
        'description': 'Loss on 50 different CIFAR-10 test samples as specified in [Jor24a]',
        'paper_reference': 'Jor24a',
        'dataset': 'CIFAR-10',
        'measurement_type': 'cross_entropy_loss'
    }


def validate_measurement_function_setup(test_dataset_size: int) -> None:
    """
    Validate that the measurement function setup is compatible with the test dataset.
    
    Args:
        test_dataset_size: Size of the test dataset
        
    Raises:
        ValueError: If the measurement function targets are incompatible with dataset
    """
    target_indices = get_measurement_function_targets()
    max_target_idx = max(target_indices)
    
    if max_target_idx >= test_dataset_size:
        raise ValueError(
            f"Maximum target index ({max_target_idx}) exceeds test dataset size ({test_dataset_size}). "
            f"CIFAR-10 test set should have 10,000 samples, but got {test_dataset_size}."
        )
    
    if len(target_indices) != PAPER_NUM_MEASUREMENT_FUNCTIONS:
        raise ValueError(
            f"Expected {PAPER_NUM_MEASUREMENT_FUNCTIONS} measurement functions, "
            f"but got {len(target_indices)} target indices."
        )

def save_json_log_entry(log_entry: Dict[str, Any], log_file_path: Path, is_json_lines: bool = False) -> None:
    """
    Saves a single log entry to a JSON or JSON Lines file.

    Args:
        log_entry: The dictionary to save as a JSON entry.
        log_file_path: Path to the log file.
        is_json_lines: If True, appends as a new line in JSONL format.
                       If False, assumes the file contains a list of entries and appends to that list.
    """
    logger = logging.getLogger('influence_analysis.utils')
    try:
        log_file_path.parent.mkdir(parents=True, exist_ok=True)

        if is_json_lines:
            with open(log_file_path, 'a') as f:
                json.dump(log_entry, f, default=str)
                f.write('\n')
        else:
            existing_logs = []
            if log_file_path.exists() and log_file_path.stat().st_size > 0:
                try:
                    with open(log_file_path, 'r') as f:
                        content = f.read()
                        if content.strip(): # Ensure content is not just whitespace
                            data = json.loads(content)
                            if isinstance(data, list):
                                existing_logs = data
                            else:
                                # If it's not a list (e.g., old format or single dict), wrap it
                                existing_logs = [data]
                                logger.warning(f"Log file {log_file_path} was not a list. Wrapped existing content.")
                except json.JSONDecodeError as e:
                    logger.error(f"Error decoding JSON from {log_file_path}: {e}. Log file might be corrupted. Backing up and starting new log.")
                    backup_path = log_file_path.with_suffix(f".{datetime.now().strftime('%Y%m%d%H%M%S')}.bak")
                    try:
                        log_file_path.rename(backup_path)
                        logger.info(f"Backed up corrupted log to {backup_path}")
                    except OSError as backup_e:
                        logger.error(f"Could not back up corrupted log {log_file_path}: {backup_e}")
                    existing_logs = [] # Start fresh
            
            existing_logs.append(log_entry)
            with open(log_file_path, 'w') as f:
                json.dump(existing_logs, f, indent=2, default=str)
        
        logger.debug(f"Saved log entry to {log_file_path} (JSON Lines: {is_json_lines})")

    except Exception as e:
        logger.error(f"Failed to save log entry to {log_file_path}: {e}")

def save_training_metrics(metrics_data: Dict[str, Any], 
                         log_file_path: Path,
                         stage: str = "training", 
                         model_id: Optional[int] = None) -> None:
    """
    Unified function to save training metrics and hyperparameters to disk as a list of JSON entries.
    
    Args:
        metrics_data: Dictionary containing metrics to save
        log_file_path: Path to the log file where metrics should be saved
        stage: Stage of training ("training", "validation", "config", etc.)
        model_id: Optional model identifier (used by LDS for multi-model training)
    """
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "stage": stage,
        "data": metrics_data
    }
    if model_id is not None:
        log_entry["model_id"] = model_id
    
    save_json_log_entry(log_entry, log_file_path, is_json_lines=False)

def prepare_optimizer_param_groups(
    model: torch.nn.Module,
    base_lr: float,
    weight_decay: float,
    bias_lr_scale: float, # e.g., config.RESNET9_BIAS_SCALE
    custom_lr_rules: Optional[Dict[str, float]] = None # For future more complex rules
) -> List[Dict[str, Any]]:
    """
    Prepares parameter groups for an optimizer with specific LR and weight decay rules.
    Commonly used to apply no weight decay to biases and batchnorm parameters,
    and potentially scale their learning rates.

    Args:
        model: The model whose parameters are to be grouped.
        base_lr: The base learning rate for most parameters.
        weight_decay: The weight decay for parameters in the 'decay' group.
        bias_lr_scale: Multiplier for the learning rate of bias/BN parameters.
        custom_lr_rules: Optional dictionary for more fine-grained LR rules (not used yet).

    Returns:
        A list of dictionaries, where each dictionary defines a parameter group.
    """
    params_decay = []
    params_no_decay_scaled_lr = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # Check if parameter should be in the no_decay_scaled_lr group
        # Criteria: 1D tensor (often biases, 1D BN params), or name ends with ".bias",
        # or "bn" (batchnorm) or "norm" (layernorm, etc.) is in the name.
        if len(param.shape) == 1 or name.endswith(".bias") or "bn" in name.lower() or "norm" in name.lower():
            params_no_decay_scaled_lr.append(param)
        else:
            params_decay.append(param)

    optimizer_grouped_parameters = [
        {'params': params_decay, 'weight_decay': weight_decay, 'lr': base_lr},
        {'params': params_no_decay_scaled_lr, 'weight_decay': 0.0, 'lr': base_lr * bias_lr_scale}
    ]
    
    # Log the grouping for clarity
    num_decay = sum(p.numel() for p in params_decay)
    num_no_decay = sum(p.numel() for p in params_no_decay_scaled_lr)
    logging.getLogger('influence_analysis.utils').debug(
        f"Optimizer param groups: {num_decay} params with WD={weight_decay}, LR={base_lr}; "
        f"{num_no_decay} params with WD=0.0, LR={base_lr * bias_lr_scale}"
    )
            
    return optimizer_grouped_parameters

def create_primary_training_optimizer(
    model: torch.nn.Module,
    master_seed: int,
    instance_id: str, # e.g., SHARED_OPTIMIZER_INSTANCE_ID or a model-specific ID for LDS
    optimizer_type_config: str, # Typically 'SGD' for this project
    base_lr_config: float,
    momentum_config: float,
    weight_decay_config: float,
    nesterov_config: bool,
    bias_lr_scale_config: float, # e.g., config.RESNET9_BIAS_SCALE
    component_logger: logging.Logger # Logger from the calling component (MagicAnalyzer/LDS)
) -> torch.optim.Optimizer:
    """
    Creates the primary training optimizer (e.g., SGD) using shared logic for
    parameter grouping and deterministic instantiation.

    Args:
        model: The model to optimize.
        master_seed: The global master seed.
        instance_id: A unique ID for this optimizer instance for seed derivation.
        optimizer_type_config: String identifier for the optimizer type (e.g., "SGD").
        base_lr_config: Base learning rate from global config.
        momentum_config: Momentum from global config.
        weight_decay_config: Weight decay for 'decay' params from global config.
        nesterov_config: Nesterov flag from global config.
        bias_lr_scale_config: LR scale factor for bias/BN params from global config.
        component_logger: Logger of the component requesting the optimizer.

    Returns:
        An instantiated PyTorch optimizer.
    """
    if optimizer_type_config.lower() != 'sgd':
        component_logger.warning(
            f"Requested optimizer type is '{optimizer_type_config}', but this utility "
            f"is primarily configured for SGD based on project structure. Proceeding with SGD."
        )
    
    optimizer_grouped_parameters = prepare_optimizer_param_groups(
        model=model,
        base_lr=base_lr_config,
        weight_decay=weight_decay_config,
        bias_lr_scale=bias_lr_scale_config
    )
    
    component_logger.info(
        f"Creating SGD optimizer with base LR={base_lr_config}, Momentum={momentum_config}, "
        f"Nesterov={nesterov_config}. Bias/BN LR scaled by: {bias_lr_scale_config}. "
        f"Weight Decay for decay group: {weight_decay_config}, for no_decay group: 0.0."
    )

    optimizer = create_deterministic_optimizer(
        master_seed=master_seed,
        optimizer_class=torch.optim.SGD,
        model_params=optimizer_grouped_parameters,
        instance_id=instance_id,
        momentum=momentum_config,
        nesterov=nesterov_config
        # lr and weight_decay are now handled by the groups
    )
    return optimizer