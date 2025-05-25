import random
import numpy as np
import torch
import logging
import sys
from typing import Optional, Any, Callable, Union
from contextlib import contextmanager
import hashlib

from . import config # Import the config module

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
    Create a model with deterministic initialization.
    
    Args:
        master_seed: Master seed
        creator_func: Function that creates the model
        instance_id: Instance identifier. Used to differentiate seeds for different
                     model instances or ensure identical seeds for shared instances.
        **kwargs: Arguments to pass to creator_func
    
    Returns:
        Created model
    """
    logger = logging.getLogger('influence_analysis.deterministic')
    log_instance_name = f"instance: {instance_id}" if instance_id else "default instance"
    logger.debug(f"Creating deterministic model ({log_instance_name})")
    
    # Use a fixed "model" string for the type part of seed derivation.
    # instance_id provides user-defined differentiation.
    component_seed = derive_component_seed(master_seed, "model", instance_id)
    
    context_name = f"model_creation ({log_instance_name})"
    with deterministic_context(component_seed, context_name):
        model = creator_func(**kwargs)
    
    logger.info(f"Created deterministic model ({log_instance_name}) with component seed {component_seed}")
    return model

def create_deterministic_optimizer(
    master_seed: int, 
    optimizer_class: type, 
    model_params, 
    instance_id: Optional[Union[str, int]] = None, 
    **kwargs
) -> Any:
    """
    Create an optimizer with deterministic initialization.
    
    Args:
        master_seed: Master seed
        optimizer_class: Optimizer class (e.g., torch.optim.SGD)
        model_params: Model parameters to optimize
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
        scheduler = create_scheduler(optimizer, schedule_type, total_steps, **scheduler_params)
    
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

def log_scheduler_info(scheduler: Optional[torch.optim.lr_scheduler._LRScheduler], 
                      schedule_type: Optional[str], logger: logging.Logger, 
                      component_name: str = "component") -> None:
    """
    Log scheduler configuration information.
    
    Args:
        scheduler: The scheduler instance
        schedule_type: Type of scheduler
        logger: Logger instance
        component_name: Name of component using scheduler (for logging)
    """
    if scheduler is None:
        logger.debug(f"{component_name}: No learning rate scheduler used")
    elif schedule_type == 'StepLR':
        logger.debug(f"{component_name}: Using StepLR scheduler: step_size={scheduler.step_size}, gamma={scheduler.gamma}")
    elif schedule_type == 'CosineAnnealingLR':
        logger.debug(f"{component_name}: Using CosineAnnealingLR scheduler: T_max={scheduler.T_max}")
    elif schedule_type == 'OneCycleLR':
        # Log the configured max_lr for OneCycleLR, as inspecting internal attributes like .max_lrs can be tricky
        logger.debug(f"{component_name}: Using OneCycleLR scheduler: configured max_lr={config.ONECYCLE_MAX_LR}, total_steps={scheduler.total_steps}")
    else:
        logger.debug(f"{component_name}: Using {schedule_type} scheduler")


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
# IMPROVED COMPONENT CREATION FUNCTIONS
# ===================================== 