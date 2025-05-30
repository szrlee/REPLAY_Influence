# ⚙️ Determinism Strategy for Reproducible Research

**Document Version**: 1.0.0
**Date**: Project Update

---

This document outlines the strategies and components implemented to achieve a high degree of reproducibility in the training and analysis pipelines, particularly for the REPLAY algorithm. Ensuring determinism is crucial for validating results, debugging, and enabling reliable research.

## 1. Core Principles of Determinism

Our approach to determinism is built on several core principles:

*   **Master Seed Control**: A single master seed (`config.SEED`) is the root for all random number generation.
*   **Component-Specific Seeding**: To avoid unintended correlations (e.g., model weights being correlated with data shuffling patterns due to using the same seed), component-specific seeds are derived from the master seed.
*   **Deterministic Operations**: Wherever possible, PyTorch operations are configured to run deterministically.
*   **Consistent Initialization**: All functionally identical components (e.g., models used in different phases of analysis that should start the same) are initialized using the same derived seed and creation logic.
*   **Environment Consistency**: While harder to control fully via code, awareness of environment factors (CUDA versions, library versions) is noted. The global deterministic settings aim to mitigate some of this.

## 2. Key Determinism-Related Fixes and Enhancements

Several critical adjustments were made to the codebase to bolster determinism:

### 2.1. Model Creation Consistency
**Original Problem**: In some parts of the analysis (e.g., `temp_model_for_target_grad` in `src/magic_analyzer.py`), models were created using direct constructor calls rather than the project's deterministic creation utilities.
**Solution**: All model instances are now created using `create_deterministic_model()`. This utility takes the master seed and an `instance_id` to derive a specific seed for that model's initialization, ensuring that models intended to be identical are, and models intended to be different (but still deterministic) can be.

```python
# Example of consistent, deterministic model creation:
# temp_model_for_target_grad = create_deterministic_model(
#     master_seed=config.SEED, creator_func=construct_rn9,
#     instance_id="magic_target_gradient_model", num_classes=config.NUM_CLASSES
# ).to(config.DEVICE)
```
**Impact**: Resolved potential inconsistencies in model initializations that could break influence computation accuracy.

### 2.2. Optimizer & Scheduler Determinism
**Original Problem**: The creation of optimizers and learning rate schedulers lacked explicit seed management, potentially leading to slight variations in training dynamics if their internal states were affected by initialization order or other subtle factors.
**Solution**: Deterministic creation utilities `create_deterministic_optimizer()` and `create_deterministic_scheduler()` were introduced. These functions also use the `master_seed` and an `instance_id` to ensure that optimizers and schedulers are initialized reproducibly.

```python
# Conceptual Signatures:
# def create_deterministic_optimizer(master_seed, optimizer_class, model_params, instance_id, **kwargs)
# def create_deterministic_scheduler(master_seed, optimizer, schedule_type, total_steps, instance_id, **scheduler_params)
```
**Impact**: Ensures highly reproducible training outcomes by managing the initialization of these crucial components.

### 2.3. DataLoader Worker Determinism
**Original Problem**: Using multiple DataLoader workers (`num_workers > 0`) can introduce non-determinism in the data loading order if the workers are not seeded correctly. Each worker process might have its own RNG state.
**Solution**: The `create_deterministic_dataloader()` utility in `src/utils.py` was enhanced. It now employs:
    *   A `DeterministicSampler` that ensures a consistent shuffling or iteration order based on a derived seed.
    *   A `seed_worker` function for the `worker_init_fn` argument of `DataLoader`. This function uses `torch.initial_seed()` (which is influenced by the generator passed to the DataLoader) to seed each worker process's RNGs (Python's `random`, `numpy.random`, and `torch`).
    *   A `torch.Generator` object, seeded with a derived component seed, passed to the `DataLoader`.

```python
# Simplified conceptual flow from create_deterministic_dataloader:
# component_seed = derive_component_seed(master_seed, "dataloader", instance_id)
# sampler = DeterministicSampler(len(dataset), shuffle=True, seed=component_seed)
# generator = torch.Generator().manual_seed(component_seed)
# deterministic_loader = DataLoader(
#     dataset,
#     sampler=sampler,
#     worker_init_fn=seed_worker, # Ensures each worker is seeded based on a master plan
#     generator=generator,        # Controls initial seed for workers and sampling if no sampler
#     # ... other args ...
# )
```
**Impact**: Ensures highly deterministic data loading sequences across runs, even when `num_workers > 0`. Setting `num_workers=0` is an alternative that also achieves determinism by using the main process for data loading.

## 3. Seed Management System

A robust seed management system is central to our determinism strategy.

### 3.1. Component-Specific Seed Derivation

To prevent hidden correlations from using a single seed for all random processes, we derive component-specific seeds from the `master_seed` using a combination of the purpose and an optional instance identifier.

```python
import hashlib
from typing import Optional, Union

def derive_component_seed(master_seed: int, purpose: str, instance_id: Optional[Union[str, int]] = None) -> int:
    \"\"\"
    Derive component-specific seeds to avoid correlations using SHA256.
    SHA256 is used for its better distribution properties compared to Python's
    built-in hash() for this kind of derivation, and for cross-platform consistency.
    The derived seed is modulo 2**31 to fit within typical integer limits for seeds.
    \"\"\"
    seed_string = f\"{master_seed}_{purpose}\"
    if instance_id is not None:
        seed_string += f\"_{instance_id}\"
    
    hash_object = hashlib.sha256(seed_string.encode())
    # Take first 8 hex digits (32 bits) and convert to int, then ensure positive
    derived_seed = int(hash_object.hexdigest()[:8], 16) % (2**31) 
    return derived_seed
```

**Benefits**:
*   **Reduces Risk of Correlations**: For example, model weight initialization is decorrelated from data shuffling order.
*   **Deterministic Derivation**: The same `master_seed`, `purpose`, and `instance_id` will always produce the same component seed.
*   **Cross-Platform Consistency**: `hashlib.sha256` provides consistent hashing results across different platforms, unlike Python's built-in `hash()` which can vary.
*   **Instance-Specific Seeds**: Allows for creating multiple deterministic instances of a component (e.g., different models in an ensemble, or models for LDS vs. MAGIC phases) that are each internally consistent but different from each other if desired (by using different `instance_id`s).

### 3.2. Global Deterministic State Setup

A utility function `set_global_deterministic_state()` is called at the beginning of main scripts to configure PyTorch and other libraries for deterministic behavior.

```python
import random
import numpy as np
import torch
import os
# from src.utils import logger # Assuming logger is available

def set_global_deterministic_state(master_seed: int, enable_deterministic: bool = True) -> None:
    \"\"\"Set global deterministic state based on PyTorch 2.2+ best practices.\"\"\"
    # Set all library seeds
    random.seed(master_seed)
    np.random.seed(master_seed)
    torch.manual_seed(master_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(master_seed) # for all GPUs
    
    if enable_deterministic:
        # CUBLAS workspace configuration:
        # Required for deterministic convolution algorithms on CUDA.
        # ':4096:8' indicates 4MB workspace for 4096-byte alignment and 8MB for 8-byte alignment.
        # These values are common defaults.
        if 'CUBLAS_WORKSPACE_CONFIG' not in os.environ:
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' # Or ':16:8' for less memory
        
        # Core PyTorch deterministic settings
        torch.use_deterministic_algorithms(True, warn_only=True) # Use 'warn_only=True' to catch ops without deterministic alternatives
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False # Must be False for determinism
        
        # Fill uninitialized memory (helps with determinism on some versions/ops)
        if hasattr(torch.utils.deterministic, 'fill_uninitialized_memory'):
            torch.utils.deterministic.fill_uninitialized_memory = True
        
        # logger.info(f"Global deterministic state set with master_seed: {master_seed}")
    # else:
        # logger.info(f"Global deterministic state NOT fully enabled. RNGs seeded with {master_seed}, but PyTorch deterministic algorithms OFF.")
```
**Note**: Full determinism, especially across different hardware or CUDA versions, can be challenging. `torch.use_deterministic_algorithms(True, warn_only=True)` will raise errors if an operation lacks a deterministic implementation, helping to identify such cases.

### 3.3. Deterministic Context Management (Optional)

For highly localized operations where you want to ensure a specific RNG state without affecting the global RNG sequence, a context manager can be useful. (Note: This is presented as a utility; its widespread use depends on specific needs.)

```python
from contextlib import contextmanager
# from src.utils import logger # Assuming logger is available

@contextmanager
def deterministic_context(component_seed: int, component_name: str = "operation"):
    \"\"\"
    Lightweight context manager to set a deterministic seed for a block of code,
    preserving and restoring the existing global RNG state of PyTorch.
    \"\"\"
    # Save current PyTorch RNG state
    old_rng_state = torch.get_rng_state()
    old_cuda_rng_states = None
    
    if torch.cuda.is_available():
        try:
            old_cuda_rng_states = torch.cuda.get_rng_state_all()
        except RuntimeError as e:
            # logger.warning(f"Could not save all CUDA RNG states for context '{component_name}': {e}")
            pass # Continue if CUDA state saving fails for some reason (e.g. no CUDA devices initialized)

    # Set new deterministic state for the context
    torch.manual_seed(component_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(component_seed)
    
    # logger.debug(f"Entering deterministic context '{component_name}' with seed {component_seed}")
    
    try:
        yield
    finally:
        # Restore previous PyTorch RNG state
        torch.set_rng_state(old_rng_state)
        if torch.cuda.is_available() and old_cuda_rng_states is not None:
            try:
                torch.cuda.set_rng_state_all(old_cuda_rng_states)
            except RuntimeError as e:
                # logger.warning(f"Could not restore all CUDA RNG states for context '{component_name}': {e}")
                pass
        # logger.debug(f"Exiting deterministic context '{component_name}'")

```
This context manager ensures that operations within its scope use a dedicated seed, and then restores the previous RNG state, preventing side effects on other parts of the code.

### 3.4. DataLoader Worker Initialization (`seed_worker`)

As mentioned in Fix #2.3, the `seed_worker` function is crucial for DataLoaders with `num_workers > 0`.

```python
import random
import numpy as np
import torch

def seed_worker(worker_id: int) -> None:
    \"\"\"
    Worker initialization function for torch.utils.data.DataLoader.
    Ensures that each worker has a unique, yet deterministically derived, seed.
    The `torch.initial_seed()` within a worker is derived from the master seed
    passed to the DataLoader's generator.
    \"\"\"
    worker_seed = torch.initial_seed() % 2**32 # Get seed set by DataLoader for this worker
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    # Potentially add:
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(worker_seed) # If workers use CUDA, though unusual for typical data loading
```
This function ensures that each data loading worker initializes its internal random number generators (Python, NumPy, PyTorch) with a seed that is itself deterministically set by the main process via the DataLoader's `generator`.

## 4. High-Level Consistency for Comparative Analyses (e.g., MAGIC vs. LDS)

For analyses that compare different methods (like MAGIC replay vs. LDS), it's often vital that components common to both methods (e.g., the initial model state, the dataset sequence) are identical. This is achieved by using **shared instance IDs** when creating these components.

### 4.1. Shared Instance IDs

By using the same `instance_id` string when calling deterministic creation utilities for components that need to be identical, we ensure they are initialized from the same derived seed.

```python
# Example from config.py or setup:
# config.SHARED_MODEL_INSTANCE_ID = "shared_base_training_model"
# config.SHARED_DATALOADER_INSTANCE_ID = "shared_training_data_sequence"

# --- In MAGIC analysis setup ---
# magic_model = create_deterministic_model(
#     master_seed=config.SEED,
#     creator_func=construct_my_model,
#     instance_id=config.SHARED_MODEL_INSTANCE_ID 
# )
# magic_train_loader = create_deterministic_dataloader(
#     master_seed=config.SEED,
#     dataset=train_dataset,
#     instance_id=config.SHARED_DATALOADER_INSTANCE_ID,
#     # ... other DataLoader args
# )

# --- In LDS analysis setup (or another comparative setup) ---
# lds_model = create_deterministic_model( # Same master_seed, creator_func, and instance_id
#     master_seed=config.SEED,
#     creator_func=construct_my_model, 
#     instance_id=config.SHARED_MODEL_INSTANCE_ID 
# )
# lds_train_loader = create_deterministic_dataloader( # Same master_seed, dataset, and instance_id
#     master_seed=config.SEED,
#     dataset=train_dataset, 
#     instance_id=config.SHARED_DATALOADER_INSTANCE_ID,
#     # ... other DataLoader args
# )

# Result: magic_model and lds_model will have identical initial weights.
# magic_train_loader and lds_train_loader will produce identical sequences of batches.
```
This careful use of shared instance IDs within the `derive_component_seed` mechanism is fundamental for fair and reliable comparisons between different experimental branches or analyses that share common starting points or data pathways.

---

By combining these strategies—global settings, component-specific seeding, careful utility design, and consistent component instantiation—the project aims to provide a robust foundation for reproducible research. 