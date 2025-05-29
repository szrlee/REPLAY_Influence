# ResNet-9 Implementation Guide

## Quick Start

```python
from src.model_def import construct_resnet9_paper # Example model
from src.utils import create_deterministic_model
from src import config # To access config.NUM_CLASSES

# Create a ResNet-9 model (e.g., paper-compliant variant)
model = construct_resnet9_paper(num_classes=config.NUM_CLASSES)

# Or create deterministically for reproducible experiments
deterministic_model = create_deterministic_model(
    master_seed=config.SEED,
    creator_func=construct_resnet9_paper, # or any other model creator from model_def.py
    instance_id="my_experiment_model",
    num_classes=config.NUM_CLASSES # Ensure num_classes is passed if required by creator_func
)
```

## Architecture Overview

The project provides several ResNet-9 variants in `src/model_def.py`. The default model used by the training pipeline is specified by `MODEL_CREATOR_FUNCTION` in `src/config.py` (currently `construct_rn9`).

| Model Function              | Approx. Params | Key Characteristics                                       |
|-----------------------------|----------------|-----------------------------------------------------------|
| `construct_rn9()`           | ~2.3M          | A common ResNet-9 variant. **Current default in config.** |
| `construct_resnet9_paper()` | ~14.2M         | Aims to follow [Jor24a] paper specs with specific scaling.|
| `ResNet9TableArch()`        | ~14.2M         | Another interpretation based on a table specification.    |
| `make_airbench94_adapted()` | Variable       | A variant adapted from AirBench94.                        |

### Example: `construct_resnet9_paper` Architecture
This variant is designed based on specifications from [Jor24a] Table 1.

```python
# Relevant parameters from src/config.py for this model variant:
# RESNET9_WIDTH_MULTIPLIER = 2.5
# RESNET9_BIAS_SCALE = 1.0 (was 8.0, then 2.0, now 1.0 for stability)
# RESNET9_FINAL_LAYER_SCALE = 0.04
# RESNET9_POOLING_EPSILON = 0.1

def construct_resnet9_paper(
    num_classes: int = 10, # As per config.NUM_CLASSES
    # Default args below match those in config.py for this specific model
    width_multiplier: float = 2.5, # from config.RESNET9_WIDTH_MULTIPLIER
    bias_scale: float = 1.0,       # from config.RESNET9_BIAS_SCALE
    final_layer_scale: float = 0.04, # from config.RESNET9_FINAL_LAYER_SCALE
    pooling_epsilon: float = 0.1   # from config.RESNET9_POOLING_EPSILON
) -> torch.nn.Module
```

**Conceptual Architecture (Channels after width_multiplier=2.5):**
- Conv1: 3 → 160 (64*2.5), 3x3 conv + BatchNorm + ReLU
- Conv2: 160 → 320 (128*2.5), 3x3 conv + BatchNorm + ReLU + LogSumExpPool2d (stride 2)
- Residual Block 1: 320 → 320
- Conv3: 320 → 640 (256*2.5), 3x3 conv + BatchNorm + ReLU
- LogSumExp Pooling (kernel 2, stride 2)
- Residual Block 2: 640 → 640
- Conv4: 640 → 320 (128*2.5), 3x3 conv (padding 0) + BatchNorm + ReLU
- Global Adaptive Average Pooling
- Classifier: Flatten → Linear (320 → num_classes) → Scale by `final_layer_scale`

## Parameter Grouping Strategy

ResNet-9 training often benefits from specialized parameter groups for the optimizer:

```python
# Illustrative example of how parameter groups might be set up.
# Actual grouping is handled in src/magic_analyzer.py and src/lds_validator.py
# using values from src/config.py.

def get_example_parameter_groups(model, effective_lr, weight_decay, bias_lr_scale):
    """
    Group 1 (Decay): Convolutional and linear weights
    - Apply weight decay (e.g., config.MODEL_TRAIN_WEIGHT_DECAY)
    - Use base learning rate (e.g., config.MODEL_TRAIN_LR)
    
    Group 2 (No-decay): Biases and BatchNorm parameters
    - No weight decay (0.0)
    - Scaled learning rate (e.g., config.MODEL_TRAIN_LR * config.RESNET9_BIAS_SCALE)
    """
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_bias_or_bn = (
            len(param.shape) == 1 or
            name.endswith(".bias") or
            "bn" in name or
            "norm" in name.lower()
        )
        if is_bias_or_bn:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    return [
        {
            'params': decay_params,
            'weight_decay': weight_decay, # e.g., 0.001 from config
            'lr': effective_lr          # e.g., 0.025 from config
        },
        {
            'params': no_decay_params,
            'weight_decay': 0.0,
            'lr': effective_lr * bias_lr_scale # e.g., 0.025 * 1.0 from config
        }
    ]
```

**Rationale:**
- BatchNorm parameters and biases often benefit from different/no weight decay and potentially different learning rates.

## Training Configuration (from `src/config.py`)

Key hyperparameters (refer to `src/config.py` for current values):

```python
# Core training parameters (see src/config.py for authoritative values)
MODEL_TRAIN_LR = 0.025            # Peak learning rate for OneCycleLR
MODEL_TRAIN_EPOCHS = 1            # Example: Number of training epochs
MODEL_TRAIN_BATCH_SIZE = 1000
MODEL_TRAIN_MOMENTUM = 0.875
MODEL_TRAIN_WEIGHT_DECAY = 0.001  # For parameters subject to decay
MODEL_TRAIN_NESTEROV = True

# Architecture-specific parameters for some model variants (see src/config.py)
RESNET9_WIDTH_MULTIPLIER = 2.5
RESNET9_BIAS_SCALE = 1.0          # Affects bias/BN LR scaling if used by optimizer setup
RESNET9_FINAL_LAYER_SCALE = 0.04
RESNET9_POOLING_EPSILON = 0.1

# Learning rate schedule (see src/config.py)
LR_SCHEDULE_TYPE = 'OneCycleLR'
ONECYCLE_MAX_LR = MODEL_TRAIN_LR # Typically set to MODEL_TRAIN_LR
ONECYCLE_PCT_START = 0.5
ONECYCLE_ANNEAL_STRATEGY = 'linear'
ONECYCLE_DIV_FACTOR = 1.0 / 0.07
ONECYCLE_FINAL_DIV_FACTOR = 0.35
```

## Usage Examples

### Basic Model Creation

```python
# Use configured model from src/config.py
from src import config
from src.utils import create_deterministic_model

# This uses the MODEL_CREATOR_FUNCTION defined in config.py
model = create_deterministic_model(
    master_seed=config.SEED,
    creator_func=config.MODEL_CREATOR_FUNCTION,
    instance_id="my_default_model",
    num_classes=config.NUM_CLASSES
)

# Direct creation of a specific variant with parameters
from src.model_def import construct_resnet9_paper
model_paper_variant = construct_resnet9_paper(
    num_classes=config.NUM_CLASSES,
    width_multiplier=config.RESNET9_WIDTH_MULTIPLIER,
    bias_scale=config.RESNET9_BIAS_SCALE # These come from config
)
```

### Integration with MAGIC Analysis (Conceptual)
```python
from src.magic_analyzer import MagicAnalyzer

# Analyzer uses the model and training settings from src/config.py
analyzer = MagicAnalyzer() # main_runner.py currently sets use_memory_efficient_replay=False
# total_steps = analyzer.train_and_collect_intermediate_states()
# scores = analyzer.compute_influence_scores(total_steps)
# analyzer.plot_magic_influences(scores) # If scores are successfully computed
```

### Measurement Functions Example
```python
from src.utils import evaluate_measurement_functions, get_measurement_function_targets
from src.data_handling import CustomDataset
from src import config # For config.CIFAR_ROOT and config.NUM_CLASSES
from src.model_def import construct_resnet9_paper # Example model

model = construct_resnet9_paper(num_classes=config.NUM_CLASSES)
# Assume model is trained or loaded

# Evaluate 50 measurement functions (φᵢ) from paper
test_dataset = CustomDataset(root=config.CIFAR_ROOT, train=False, download=True)
targets = get_measurement_function_targets()  # [0, 1, ..., 49]
# losses = evaluate_measurement_functions(model, test_dataset, targets)
# print(f"φ₀(θ) = {losses[0]:.4f}")  # Loss on test sample 0
```

## Configuration & Deployment

### Model Selection (in `src/config.py`)
```python
# Example: To use construct_resnet9_paper as the default for training
# MODEL_CREATOR_FUNCTION = construct_resnet9_paper
# Current default in config.py is construct_rn9
MODEL_CREATOR_FUNCTION = construct_rn9 
```

### Command Line Usage (via `main_runner.py`)
(Refer to main project README.md for detailed CLI options)
```bash
# Run MAGIC analysis using the model specified in src/config.py
python main_runner.py --magic

# Run LDS validation 
python main_runner.py --lds --run_id <your_magic_run_id>

# Check parameter count of the default configured model
python -c "
from src import config
from src.utils import create_deterministic_model
model = create_deterministic_model(config.SEED, config.MODEL_CREATOR_FUNCTION, \"count_params\")
params = sum(p.numel() for p in model.parameters())
print(f'{config.MODEL_CREATOR_FUNCTION.__name__} Parameters: {params:,}')
"
```

## Performance Characteristics (Illustrative)

| Model Function Example      | Approx. Params | Notes                                                    |
|-----------------------------|----------------|----------------------------------------------------------|
| `construct_rn9()`           | ~2.3M          | Simpler, faster variant. Current default in `config.py`. |
| `construct_resnet9_paper()` | ~14.2M         | Aims for paper specs, more complex.                      |
| `ResNet9TableArch()`        | ~14.2M         | Alternative complex variant.                             |
*Performance (memory/speed) varies greatly with hardware and specific configuration.*