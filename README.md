# Influence Analysis Project (MAGIC & LDS)

## Overview

This project implements and explores methods for understanding how individual training data points affect a model's predictions and internal states. It focuses on:

1.  **Influence Function Calculation (MAGIC Analysis):** This involves computing influence scores for training samples, similar to methods like TracIn/REPLAY. These scores help identify which training examples were most responsible for a model's behavior on a specific test (or validation) instance. This is invaluable for model debugging, understanding dataset biases, and identifying influential or potentially mislabeled training data.
2.  **Validation of Influence Scores (LDS Validation):** A method to empirically validate the computed influence scores by training multiple models on systematically generated subsets of the training data. The performance of these subset-trained models on a target instance is then correlated with the influence scores, providing a sanity check for the influence estimation.

The project uses the CIFAR-10 dataset and a ResNet9 model architecture as a concrete testbed for these analyses.

## Key Features & Recent Improvements

### 🔧 **Algorithmic Correctness & Data Consistency**
- **Data Ordering Verification**: Comprehensive 6-test verification system ensures MAGIC and LDS use identical data sequences
- **PyTorch DataLoader Determinism**: Fixes for multi-dataloader random state management ensuring reproducible results
- **Model Initialization Consistency**: All LDS models start with identical weights, differing only in training subsets
- **Configuration Validation**: Automatic validation prevents parameter mismatches that could invalidate results

### 🚀 **Memory & Performance Optimizations**
- **Memory-Efficient Batch Replay**: Stream batch data from disk during replay, reducing memory usage by ~80%
- **Optimized DataLoader Settings**: Consistent `num_workers` parameters across all dataloaders for determinism
- **Smart Caching**: Automatic detection of existing results to avoid redundant computation
- **Progress Tracking**: Detailed logging and progress bars for long-running computations

### 🧪 **Robustness & Validation**
- **Comprehensive Testing**: Built-in verification functions validate data consistency before analysis
- **Error Handling**: Graceful handling of edge cases with informative error messages
- **Configuration Consistency**: Automatic checks ensure MAGIC and LDS target the same validation image
- **Reproducibility**: Fixed seed management ensures identical results across runs

## Project Structure

The project is organized as follows:

```
influence_project/
├── src/                    # Source code for the analyses
│   ├── __init__.py         # Package initializer
│   ├── config.py           # Shared configurations, hyperparameters, and paths
│   ├── data_handling.py    # CIFAR-10 dataset loading and preprocessing logic
│   ├── model_def.py        # ResNet9 model architecture definition
│   ├── magic_analyzer.py   # Core implementation of MAGIC influence calculation (TracIn/REPLAY-like)
│   ├── lds_validator.py    # Core implementation of LDS subset training and validation
│   ├── utils.py            # Utility functions for seeding and logging
│   └── visualization.py    # Plotting functions for visualizing influence and correlation
├── main_runner.py          # Main script to execute MAGIC and/or LDS analyses
├── outputs/                # Directory for all generated files
│   ├── checkpoints_magic/  # Model checkpoints saved during MAGIC model training (one per step)
│   ├── checkpoints_lds/    # Final model checkpoints for each LDS subset model
│   ├── scores_magic/       # Pickled NumPy arrays of computed MAGIC influence scores
│   ├── losses_lds/         # Pickled NumPy arrays of per-sample validation losses for each LDS model
│   ├── plots_magic/        # MAGIC analysis plots (influential images)
│   └── plots_lds/          # LDS validation plots (correlation analysis)
├── data/                   # (Currently unused, CIFAR-10 downloads to /tmp/cifar/ by default via config.py)
├── notebooks/              # (Optional, for experimental code and exploration)
├── README.md               # This file
└── requirements.txt        # Python dependencies
```

## Requirements

*   Python 3.8+
*   PyTorch (see `requirements.txt` for version, e.g., 1.10+)
*   NumPy
*   Matplotlib
*   Seaborn
*   tqdm
*   SciPy

For a full list of dependencies and their versions, please refer to `requirements.txt`.

## Setup

1.  **Create a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Linux/macOS
    # venv\Scripts\activate    # On Windows
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Running the Analyses

The primary entry point for running the analyses is `main_runner.py`. You can see all available options by running:
```bash
python main_runner.py --help
```

**Workflow:**

The typical workflow involves:
1.  Running the MAGIC analysis to compute influence scores for a target validation image.
2.  Optionally, running the LDS validation, which uses the scores from the MAGIC analysis.

**Examples:**

1.  **Run only the MAGIC influence analysis:**
    ```bash
    python main_runner.py --run_magic
    ```
    This will:
    *   Automatically verify data ordering consistency before training
    *   Train a ResNet9 model on CIFAR-10 from scratch, saving model checkpoints at each training step to `outputs/checkpoints_magic/`
    *   Compute influence scores for a pre-defined target validation image (configured in `src/config.py`)
    *   Save the raw influence scores as a pickled NumPy array to `outputs/scores_magic/`
    *   Generate and save a plot of the most and least influential training images for the target image to `outputs/plots_magic/`

2.  **Run only the LDS validation (requires pre-computed MAGIC scores):**
    LDS validation relies on influence scores generated by the MAGIC analysis.
    *   If `magic_analyzer.py` was run previously and its `MAGIC_TARGET_VAL_IMAGE_IDX` matches `LDS_TARGET_VAL_IMAGE_IDX_FOR_CORRELATION` in `src/config.py`, LDS validation can find the scores file automatically.
      ```bash
      python main_runner.py --run_lds
      ```
    *   Alternatively, you can explicitly specify the path to the MAGIC scores file:
      ```bash
      python main_runner.py --run_lds --magic_scores_file outputs/scores_magic/magic_scores_per_step_val_21.pkl
      ```
    This will:
    *   Perform comprehensive data ordering verification (6 tests) to ensure consistency with MAGIC
    *   Generate definitions for training data subsets (or load them if they exist from a previous run, see `outputs/indices_lds.pkl`)
    *   Train multiple ResNet9 models with identical initialization, each on a different data subset, saving their final checkpoints to `outputs/checkpoints_lds/`
    *   Evaluate these subset-trained models on the entire validation set and save their per-sample validation losses to `outputs/losses_lds/`
    *   Correlate the performance of these subset-trained models (specifically, their loss on the target validation image) with predictions derived from the MAGIC influence scores
    *   Generate and save a correlation plot to `outputs/plots_lds/`

3.  **Run both MAGIC and LDS sequentially:**
    ```bash
    python main_runner.py --run_magic --run_lds
    ```
    This is a convenient way to ensure the MAGIC scores are generated and then immediately used by the LDS validation, especially if the target validation image for both analyses is the same (as per default configuration).

4.  **Run with memory-efficient mode (recommended for large datasets):**
    ```bash
    python main_runner.py --run_magic --memory_efficient
    ```
    This mode saves memory by streaming batch data from disk during replay instead of keeping all batches in memory. Reduces memory usage by approximately 80% with minimal performance impact.

5.  **Run with custom logging:**
    ```bash
    python main_runner.py --run_magic --log_level DEBUG --log_file analysis.log
    ```
    This enables debug logging and saves logs to a file for detailed debugging and progress tracking.

6.  **Show configuration summary:**
    ```bash
    python main_runner.py --show_config
    ```
    This displays all configuration parameters and validation results, then exits without running any analysis.

7.  **Force recomputation of existing results:**
    ```bash
    python main_runner.py --run_magic --force_retrain --force_recompute
    ```
    Use these flags to force regeneration of model checkpoints (`--force_retrain`) or influence scores (`--force_recompute`).

## Cleanup Options

The project provides options to clean up output files generated by previous analysis runs:

1.  **Clean MAGIC analysis outputs:**
    ```bash
    python main_runner.py --clean_magic
    ```
    This will remove:
    *   MAGIC model checkpoints from `outputs/checkpoints_magic/`
    *   MAGIC influence scores from `outputs/scores_magic/`
    *   MAGIC plots from `outputs/plots_magic/`
    *   The batch dictionary file used for REPLAY computation

2.  **Clean LDS validation outputs:**
    ```bash
    python main_runner.py --clean_lds
    ```
    This will remove:
    *   LDS subset model checkpoints from `outputs/checkpoints_lds/`
    *   LDS validation loss files from `outputs/losses_lds/`
    *   LDS correlation plots from `outputs/plots_lds/`
    *   LDS subset indices file (`outputs/indices_lds.pkl`)

3.  **Clean both MAGIC and LDS outputs:**
    ```bash
    python main_runner.py --clean_magic --clean_lds
    ```

4.  **Clean and then run fresh analyses:**
    ```bash
    python main_runner.py --clean_magic --clean_lds --run_magic --run_lds
    ```

## Performance and Memory Optimizations

The project includes several optimizations for better performance and memory usage:

### Memory-Efficient Batch Replay
- Use `--memory_efficient` flag to enable streaming batch data from disk
- Reduces memory usage significantly for large datasets (~80% reduction)
- Trades minimal speed for memory efficiency
- Recommended when running on systems with limited RAM
- Automatically handles compatibility checks between modes

### Data Ordering Consistency
- **Automatic Verification**: 6 comprehensive tests verify data ordering consistency between MAGIC and LDS
- **PyTorch DataLoader Fixes**: Proper seed management ensures identical data sequences across multiple dataloader creations
- **Shared DataLoader Architecture**: LDS uses a single shared dataloader to ensure all models see identical data ordering
- **Model Initialization Consistency**: All LDS models start with identical weights for fair comparison

### Logging and Monitoring
- Comprehensive logging system with configurable levels (DEBUG, INFO, WARNING, ERROR)
- Progress tracking with detailed status messages and time estimates
- Error handling with informative error messages and suggested fixes
- Optional log file output for debugging and analysis tracking
- Automatic configuration validation with clear error reporting

### Configuration Validation
- Automatic validation of all configuration parameters at startup
- Early detection of incompatible settings with helpful suggestions
- Helpful warnings for potential issues (e.g., subset size vs batch size)
- Configuration summary display for transparency
- Validation of file paths and directory structure

## Configuration

Key parameters, paths, and algorithm settings can be modified in `src/config.py`. This includes:

### Core Settings
*   **Target Validation Image Indices**: `MAGIC_TARGET_VAL_IMAGE_IDX` and `LDS_TARGET_VAL_IMAGE_IDX_FOR_CORRELATION` (must match for valid correlation)
*   **Random Seed**: `SEED` for reproducible results across all components
*   **Device Configuration**: `DEVICE` for GPU/CPU selection
*   **Data Path**: `CIFAR_ROOT` for CIFAR-10 dataset location

### MAGIC Analysis Configuration
*   **Training Hyperparameters**: Epochs, batch size, learning rate, momentum, weight decay
*   **Model Architecture**: ResNet9 with configurable number of classes
*   **Replay Parameters**: Learning rate (automatically matches training LR for consistency)
*   **Memory Mode**: Choice between memory-efficient and standard replay modes

### LDS Validation Configuration
*   **Subset Generation**: Fraction of data per subset, number of subsets to generate
*   **Model Training**: Number of models to train, training hyperparameters
*   **Evaluation**: Batch sizes and validation procedures
*   **Output Paths**: Separate directories for different types of outputs

### Recent Configuration Improvements
- **Simplified Parameters**: Removed `MAGIC_REPLAY_LEARNING_RATE` (now automatically uses `MAGIC_MODEL_TRAIN_LR` for consistency)
- **Validation Functions**: Built-in validation ensures parameter compatibility
- **Path Management**: Automatic directory creation and path validation
- **Consistent Naming**: Standardized parameter names across MAGIC and LDS components

## Data Ordering Verification System

The project includes a comprehensive verification system to ensure data ordering consistency:

### Verification Tests
1. **Basic Data Loader Consistency**: Verifies identical data ordering between MAGIC and LDS dataloaders
2. **Multi-Epoch Behavior**: Tests predictable shuffling behavior across training epochs
3. **Complete Data Sequence Consistency**: Critical test ensuring identical data sequences across the entire training process
4. **Subset Mechanism Verification**: Validates that weighted training correctly identifies subset samples
5. **Model Initialization Consistency**: Ensures all LDS models start with identical parameters
6. **Configuration Consistency**: Checks alignment of batch sizes, target images, and other critical parameters

### Automatic Execution
- Verification runs automatically before LDS validation
- Detailed logging of each test result
- Analysis stops with clear error messages if verification fails
- Helpful suggestions for fixing configuration issues

## Troubleshooting

### Common Issues and Solutions

1. **Data Ordering Verification Failed**
   ```
   Error: Data ordering mismatch between MAGIC and LDS - cannot proceed with validation
   ```
   **Solution**: Check that `MAGIC_MODEL_TRAIN_BATCH_SIZE == LDS_MODEL_TRAIN_BATCH_SIZE` and `MAGIC_TARGET_VAL_IMAGE_IDX == LDS_TARGET_VAL_IMAGE_IDX_FOR_CORRELATION` in `src/config.py`.

2. **Memory Issues During MAGIC Analysis**
   ```
   RuntimeError: CUDA out of memory
   ```
   **Solution**: Use the memory-efficient mode with `--memory_efficient` flag or reduce batch size in `src/config.py`.

3. **Missing MAGIC Scores for LDS Validation**
   ```
   FileNotFoundError: MAGIC scores file not found
   ```
   **Solution**: Run MAGIC analysis first with `--run_magic` or specify the correct path with `--magic_scores_file`.

4. **Configuration Validation Errors**
   ```
   ValueError: MAGIC_MODEL_TRAIN_BATCH_SIZE must be positive
   ```
   **Solution**: Check all parameters in `src/config.py` are properly set. Use `--show_config` to see current values.

### Debug Mode
For detailed debugging information:
```bash
python main_runner.py --run_magic --run_lds --log_level DEBUG --log_file debug.log
```

This provides extensive logging of:
- Data ordering verification steps
- Model initialization processes
- Training progress and memory usage
- File I/O operations
- Configuration validation results

## Notes

### Important Considerations
*   **CIFAR-10 Data Path**: The scripts assume CIFAR-10 will be downloaded to `/tmp/cifar/` by default. This can be changed by modifying `CIFAR_ROOT` in `src/config.py`.
*   **Memory Usage**: The project now supports both memory-efficient and standard modes. Memory-efficient mode is recommended for larger datasets or systems with limited RAM.
*   **Reproducibility**: All random operations are properly seeded to ensure reproducible results. The verification system ensures data ordering consistency across all components.
*   **File Organization**: Separate output directories for MAGIC and LDS components prevent file conflicts and make cleanup easier.

### Performance Notes
*   **Memory-Efficient Mode**: Trades ~10% speed for ~80% memory reduction during replay
*   **DataLoader Workers**: Configured for optimal balance between speed and determinism
*   **Checkpoint Management**: Automatic detection of existing checkpoints prevents redundant computation
*   **Batch Processing**: Optimized batch sizes and processing for typical hardware configurations

### Validation & Testing
*   **Comprehensive Testing**: Built-in verification ensures algorithmic correctness
*   **Error Detection**: Early detection of configuration issues prevents wasted computation
*   **Progress Monitoring**: Detailed logging helps track long-running analyses
*   **Result Validation**: Automatic checks ensure output file integrity and format correctness

### Recent Bug Fixes
*   **Fixed PyTorch DataLoader Random State Management**: Ensures identical data ordering across multiple dataloader creations
*   **Fixed Model Initialization Inconsistency**: All LDS models now start with identical weights
*   **Fixed Memory Efficiency Issues**: Eliminated redundant data storage in memory-efficient mode
*   **Fixed Configuration Parameter Redundancy**: Simplified configuration reduces potential for inconsistencies