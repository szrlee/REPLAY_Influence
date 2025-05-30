# 💾 **Memory Efficient Replay User Guide**

**A conceptual guide to memory efficient replay in the REPLAY Influence Analysis system**

---

## 🎯 **Understanding Memory Efficient Replay**

Memory efficient replay is a feature of the `MagicAnalyzer` class within the REPLAY Influence Analysis system. It allows influence analysis on large datasets without running out of memory. Instead of keeping all training data and intermediate states (like per-step momentum buffers) in RAM, it streams necessary data from disk during the replay phase of the analysis. This includes input data (images, labels), original indices, and critical optimizer states like momentum buffers from each training step, ensuring mathematical correctness is maintained.

### **Current `main_runner.py` Behavior for MAGIC Analysis**
**Important Note:** As of the current version, when you run MAGIC analysis using `python main_runner.py --magic`, `main_runner.py` instantiates the `MagicAnalyzer` class with `use_memory_efficient_replay=False`. This means it defaults to **in-memory replay**.

To enable memory-efficient replay for MAGIC analysis via `main_runner.py`, a developer would currently need to:
1.  Modify `main_runner.py` to pass `use_memory_efficient_replay=True` when instantiating `MagicAnalyzer`.
2.  Or, add a new command-line interface (CLI) flag to `main_runner.py` to control this parameter.

This guide describes how memory-efficient replay functions conceptually, how a developer can enable it programmatically by directly using `MagicAnalyzer`, and how CLI interaction might be designed if such a flag were implemented.

### **Programmatic Usage Example (Directly using `MagicAnalyzer`)**
```python
# Example of how a developer can directly control the mode:
from src.magic_analyzer import MagicAnalyzer

# To enable memory-efficient replay:
analyzer_mem_efficient = MagicAnalyzer(use_memory_efficient_replay=True)

# To use in-memory replay (current default when called by main_runner.py --magic):
analyzer_in_memory = MagicAnalyzer(use_memory_efficient_replay=False)

# Example instantiation in main_runner.py (simplified):
# if args.magic:
#     analyzer = MagicAnalyzer(use_memory_efficient_replay=False) # Current default
#     # ... rest of the magic analysis logic
```

### **When to Consider Memory Efficient Mode (Programmatically or via Modified Runner)**
- ✅ **Large datasets** (>100K samples where intermediate data for all steps is substantial).
- ✅ **Limited RAM** (e.g., <32GB system memory for very large runs where in-memory storage would fail).
- ✅ **Long training runs** (>5K training steps where the cumulative size of batch data and optimizer states becomes very large).
- ✅ **Encountering memory errors** (e.g., `RuntimeError: CUDA out of memory` or system OOM) with the default in-memory replay mode.

---

## 🚀 **Conceptual Workflow with Memory Efficiency**

This section outlines the conceptual workflow if memory-efficient replay is enabled (e.g., programmatically or through a modified `main_runner.py`).

### **Step 1: Check Your System**
```bash
# Check available memory
free -h

# Check available disk space (outputs/runs/<run_id>/checkpoints_magic/ will store batch files)
df -h
```

### **Step 2: Run Analysis (Illustrative CLI with a Hypothetical Memory Efficiency Flag)**

If `main_runner.py` were extended with a flag like `--magic_mem_efficient` to control this:
```bash
# Hypothetical usage if a flag were added:
# python main_runner.py --magic --magic_mem_efficient

# Current way (main_runner.py uses in-memory for --magic by default):
python main_runner.py --magic

# To achieve memory-efficient replay currently, one would modify main_runner.py
# or run the MagicAnalyzer programmatically as shown above.

# With force retrain (if switching modes or re-running, assuming mode is controlled):
# python main_runner.py --magic --force [--magic_mem_efficient]

# With debugging:
python main_runner.py --magic --log_level DEBUG
```

### **Step 3: Monitor Progress**
If memory-efficient mode is active (e.g., programmatically or via a modified runner):
- Log messages from `MagicAnalyzer` would indicate: `"Using memory-efficient batch replay (streaming from disk)"`.
- Batch files (e.g., `batch_1.pkl`, `batch_2.pkl`, ...) would be saved to the run's `checkpoints_magic` directory during the training phase (state collection).
- These batch files would be loaded from disk during the replay phase (influence computation).

---

## 🔧 **Configuration Aspects**

### **Programmatic Configuration (Direct `MagicAnalyzer` Usage)**
As shown previously, the mode is controlled by the `use_memory_efficient_replay` boolean parameter in the `MagicAnalyzer` constructor:
```python
from src.magic_analyzer import MagicAnalyzer

# Developer choice when instantiating MagicAnalyzer directly:
analyzer_mem_efficient = MagicAnalyzer(use_memory_efficient_replay=True)
analyzer_in_memory = MagicAnalyzer(use_memory_efficient_replay=False)

# If running programmatically:
# total_steps = analyzer.train_and_collect_intermediate_states(force_retrain=...)
# scores = analyzer.compute_influence_scores(total_steps, force_recompute=...)
```
**Note**: If you are using `main_runner.py` with the `--magic` flag, it currently sets `use_memory_efficient_replay=False` internally.

### **Key `main_runner.py` CLI Flags (Related to Re-computation and Debugging)**
These flags are relevant regardless of the memory mode but are important for managing runs:
```bash
# General force flag for re-computation/retraining
--force
  # For --magic: Forces model retraining (and thus re-collection of replay states) 
  # and score recomputation. Essential if you've changed how replay states are stored 
  # (e.g., by modifying code to switch between memory-efficient and in-memory).

# Skip training phase for an existing run (requires --run_id)
--skip_train
  # For --magic: Skips the model training part and attempts to load existing replay states.
  # Use with --force to recompute scores using existing artifacts if the replay states
  # are compatible with the current MagicAnalyzer mode (memory-efficient or in-memory).

# Debug mode (see detailed file operations and logging)
--log_level DEBUG
```

---

## 🛠️ **Troubleshooting**

### **Common Issues**

#### **Out of Memory Errors (OOM)**
```
RuntimeError: CUDA out of memory
# or general system OOM errors
```
**Considerations**:
- **With Default `main_runner.py --magic` Behavior**: This implies the default in-memory storage of all batch data (images, labels, momentum buffers per step) is exceeding available RAM or VRAM.
    - **Solution 1**: Reduce `MODEL_TRAIN_BATCH_SIZE` in `src/config.py` if the OOM is during the model's forward/backward pass within a replay step.
    - **Solution 2**: Modify `main_runner.py` to instantiate `MagicAnalyzer(use_memory_efficient_replay=True)` or use `MagicAnalyzer` programmatically with this setting. This will offload the per-step batch data to disk.
    - **Solution 3 (GPU specific for CUDA OOM)**: Try setting the environment variable `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before running, which can sometimes help with fragmentation issues.
      ```bash
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python main_runner.py --magic
      ```

- **If Memory-Efficient Mode is Programmatically Enabled and OOM Still Occurs**:
    - This suggests the OOM is likely happening *within* a single replay step (e.g., loading the model, processing a batch for gradient computation) rather than from storing all historical data.
    - Check `MODEL_TRAIN_BATCH_SIZE` as it's used for calculating gradients during replay. A large batch size here, combined with the model size, could still cause OOM even if historical data is on disk.

#### **Mode Mismatch Warnings/Errors (If Manually Managing Artifacts or Modifying `main_runner.py`)**
If `magic_batch_dict.pkl` was generated with one mode (e.g., in-memory) and then `MagicAnalyzer` is run in a different mode (e.g., memory-efficient by changing the code) on the same run artifacts without forcing re-collection of states:
```
WARNING: Loaded batch_dict is from regular mode, but current mode is memory-efficient... Will retrain.
# Or potentially errors if the structures are incompatible and retraining isn't forced.
```
**Solution**:
- Always use the `--force` flag with `main_runner.py --magic` when you have changed the underlying mechanism for how `MagicAnalyzer` stores or expects replay states (e.g., by toggling `use_memory_efficient_replay` in the code that calls `MagicAnalyzer`). This ensures training/state collection is redone consistently.
  ```bash
  python main_runner.py --magic --force
  ```

#### **Missing Batch Files (If Memory-Efficient Mode Was Programmatically Enabled and Files Are Missing)**
```
RuntimeError: Missing batch files for memory-efficient replay. Example: ... batch_X.pkl not found.
# Or FileNotFoundError during _load_batch_from_disk
```
**Solutions**:
1. Ensure the initial training/state collection phase completed successfully when memory-efficient mode was active.
2. Verify that the `outputs/runs/<run_id>/checkpoints_magic/` directory contains the expected `batch_*.pkl` files.
3. If files were deleted or the collection was incomplete, force retraining/re-collection of states:
   ```bash
   python main_runner.py --magic --force 
   # (Ensure your code is set to use memory-efficient mode if that's intended for this run)
   ```

#### **Slow Performance (If Memory-Efficient Mode is Programmatically Enabled)**
Disk I/O for reading `batch_*.pkl` files can be a bottleneck.
**Solutions**:
1. **Use an SSD**: Ensure the `outputs/` directory (or at least the specific run directory) is on a fast SSD.
2. **System Monitoring**: Use tools like `iostat` (Linux) or `iotop` to check if disk I/O is indeed the limiting factor.
3. **Reduce `DATALOADER_NUM_WORKERS` (from `src/config.py`)**: For the *initial training/state collection phase*, `DATALOADER_NUM_WORKERS` affects CPU-bound data loading. During the *replay phase* using memory-efficient mode, this setting is not directly relevant as batches are read one by one from disk by the main process. If the initial state collection is slow and I/O bound, reducing workers *might* paradoxically help if too many workers are thrashing the disk, but typically, more workers help with CPU-bound preprocessing. For replay itself, this config doesn't apply.
4. **Optimize File System/Caching**: Ensure your OS has adequate file system caching.
5. **Consider In-Memory if Feasible**: If sufficient RAM is available and disk I/O is a major bottleneck, using the in-memory mode (`use_memory_efficient_replay=False`) will be faster.

### **Debugging Commands**

#### **Check File Existence & Training Completion (for a specific run)**
```bash
# Example for run_id 'my_run_123'
RUN_ID="my_run_123" # Replace with your actual run ID

# Count batch files (relevant if memory-efficient mode was programmatically enabled for this run)
ls outputs/runs/$RUN_ID/checkpoints_magic/batch_*.pkl | wc -l 

# Verify final model checkpoint (sd_0_<total_steps>.pt) exists
# Replace <total_steps> with the expected number of training iterations
ls outputs/runs/$RUN_ID/checkpoints_magic/sd_0_<total_steps>.pt 

# Check the main batch dictionary (stores step numbers in memory-efficient mode, or full data in in-memory mode)
ls outputs/runs/$RUN_ID/magic_batch_dict.pkl
```

#### **Debug Mode Logging (Via `main_runner.py`)**
```bash
python main_runner.py --magic --log_level DEBUG --run_id <your_run_id_if_existing>
```
Look for log messages from `MagicAnalyzer` regarding:
- `"Using memory-efficient batch replay (streaming from disk)"` or `"Using in-memory batch replay (faster but memory-intensive)"`.
- File save/load operations for `batch_*.pkl` files if memory-efficient mode is active.
- Warnings about mode mismatches if `--force` was not used appropriately after changing how `MagicAnalyzer` is instantiated.

---

## ⚡ **Performance & Validation**

- **Storage**: An SSD for the `outputs/` directory is highly recommended if memory-efficient mode (with its disk I/O for `batch_*.pkl` files) is programmatically enabled and used.
- **System Configuration for Initial Training/State Collection**:
    - `DATALOADER_NUM_WORKERS` (in `src/config.py`): Affects data loading speed during the initial training phase where states are collected. Adjust based on CPU cores and I/O capabilities. Setting to `0` uses the main process, which can be useful for debugging data loading issues.
    - `MODEL_TRAIN_BATCH_SIZE` (in `src/config.py`): Impacts memory usage during both initial training and each step of the replay (as gradients are computed over a batch).
- **Monitoring (If memory-efficient mode is active and performance is a concern)**:
    - Use `free -h` to monitor RAM usage.
    - Use `iostat -x 1` or `iotop` (Linux) to monitor disk I/O activity.
- **Validation**:
    - Check logs for successful completion and absence of errors.
    - If memory-efficient replay was programmatically enabled, confirm that `batch_*.pkl` files were created in `outputs/runs/<run_id>/checkpoints_magic/` during state collection and that logs show them being loaded during replay.
    - The `magic_batch_dict.pkl` file will contain only step numbers as keys and simple dictionaries like `{'step': step_number}` as values if memory-efficient mode was active. If in-memory mode was used, it will contain the full batch data.

📚 **[Detailed Technical Info →](../technical/comprehensive-analysis.md#performance--memory)**

---

## 📋 **Best Practices**

### **Choosing the Replay Mode (Developer Consideration when using `MagicAnalyzer` directly or modifying `main_runner.py`)**

#### **Consider Enabling Memory Efficient Mode When:**
- Dataset size is large (e.g., >100K samples) and/or the number of training steps is high (e.g., >5K), leading to a very large collection of intermediate states (batch data + optimizer states for each step).
- System RAM is limited (e.g., <32GB), and you encounter OOM errors with the default in-memory replay.

#### **Prefer In-Memory Replay (Current Default for `--magic` in `main_runner.py`) When:**
- The dataset and total number of training steps are moderate, such that all intermediate states can comfortably fit in RAM.
- System RAM is ample (e.g., >64GB often suffices for typical CIFAR-10 runs with a few thousand steps).
- Maximum replay speed is critical, and disk I/O for the memory-efficient mode might become a bottleneck.

### **Storage Management**

#### **Disk Space Planning (If Memory-Efficient Mode is Programmatically Enabled)**
Estimate disk space for `batch_*.pkl` files:
- Each `batch_*.pkl` file stores: image tensors, label tensors, original indices, the learning rate for that step, and (if momentum > 0) momentum buffers for all model parameters.
- For CIFAR-10 (32x32x3 images), `MODEL_TRAIN_BATCH_SIZE = 1000`, and a ResNet-9 like model, a very rough estimate might be **5-50MB per step/batch file**, depending heavily on whether momentum buffers are stored (which depends on `MODEL_TRAIN_MOMENTUM > 0`).
    - Images: `1000 * 3 * 32 * 32 * 4 bytes (float32) ≈ 12.3 MB`
    - Labels/Indices: Small.
    - Momentum Buffers (ResNet-9 might have ~1-5M params): `~5M params * 4 bytes/param ≈ 20 MB`.
- **Example**: For 2000 steps, this could range from `10 GB` to `100 GB` if momentum buffers are large and saved per step. **It's crucial to test for your specific model and data.**

#### **Cleanup After Analysis**
- The `batch_*.pkl` files stored in `outputs/runs/<run_id>/checkpoints_magic/` are only needed if you intend to re-run `compute_influence_scores` *without* re-running the initial training/state collection, and memory-efficient mode was active.
- The `main_runner.py --clean --run_id <id> --what checkpoints` command will remove the entire `checkpoints_magic` directory, including these `batch_*.pkl` files and model checkpoints (`sd_0_*.pt`).
- If you only need the final scores (`magic_scores_*.pkl`) and plots, you can safely clean the checkpoints.

```bash
# To clean checkpoints (including any batch_*.pkl files within checkpoints_magic and model checkpoints):
python main_runner.py --clean --run_id <your_run_id> --what checkpoints 
# Or --what all to remove the entire run directory
```

---

## 🔄 **Managing Replay Modes (Developer Perspective When Modifying Code)**

If you modify `main_runner.py` or your programmatic script to change the `use_memory_efficient_replay` value for `MagicAnalyzer` between runs using the same `run_id`:

1.  **Strongly Recommended**: Use the `--force` flag with `main_runner.py --magic`. This forces `train_and_collect_intermediate_states` to run again, ensuring that `magic_batch_dict.pkl` and any associated `batch_*.pkl` files are generated consistently with the newly specified mode.
    ```bash
    # Assuming you've changed main_runner.py to use a different mode for MagicAnalyzer:
    python main_runner.py --magic --force --run_id <your_run_id_for_new_mode>
    ```
2.  **Alternative - Clean and Restart**: For a completely fresh state, delete the old run artifacts or use a new `run_id`.
    ```bash
    # Example: Clean all artifacts of an old run
    # python main_runner.py --clean --run_id <old_run_id> --what all
    # Then run with the new mode setting (after modifying the code)
    python main_runner.py --magic --run_id <new_run_id_for_new_mode>
    ```

### **Validation After Mode Change**
- Examine the `MagicAnalyzer` logs (enable with `--log_level DEBUG` in `main_runner.py`).
- Confirm the log message: `"Using memory-efficient batch replay (streaming from disk)"` or `"Using in-memory batch replay (faster but memory-intensive)"`.
- If memory-efficient: Check for `Saving batch X to disk...` and `Loading batch X from disk...` messages.
- If in-memory: The `magic_batch_dict.pkl` will be significantly larger as it contains all data.

---

## 🆘 **Emergency Procedures**

### **If Analysis Crashes Mid-Run**
1.  **Check Logs**: The primary log is usually console output. If `main_runner.py` were configured to write to a file (e.g., via `setup_logging` if a `--log_file` option was added), check that file. Otherwise, carefully review the console output for error messages.
2.  **Check Disk Space**: `df -h` (especially if memory-efficient mode was active and saving many batch files).
3.  **Check RAM Usage**: `free -h` (especially if in-memory mode was active).
4.  **Restart with Debug Mode**:
    ```bash
    python main_runner.py --magic --log_level DEBUG --run_id <relevant_run_id_if_continuing_or_new>
    ```
    If the run was interrupted, using `--force` might be necessary if artifacts are in an inconsistent state.
5.  **If Artifacts are Suspect**: If you suspect corrupted artifacts (e.g., `magic_batch_dict.pkl` or batch files), it's often best to clean and restart the problematic part or the whole run.
    ```bash
    # Clean checkpoints for the run, then try again with --force
    python main_runner.py --clean --run_id <problem_run_id> --what checkpoints
    python main_runner.py --magic --force --run_id <problem_run_id>

    # Or clean the entire run and start fresh
    # python main_runner.py --clean --run_id <problem_run_id> --what all
    # python main_runner.py --magic --run_id <problem_run_id_or_new>
    ```

### **If Running Out of Disk Space (Potentially with Memory-Efficient Mode Active)**
1.  **Check Usage**: `du -sh outputs/`. Identify the largest run directories.
2.  **Clean Artifacts**:
    *   Clean checkpoints from completed or unneeded runs:
        ```bash
        python main_runner.py --clean --run_id <old_run_id_1> --what checkpoints
        python main_runner.py --clean --run_id <old_run_id_2> --what logs losses plots scores
        ```
    *   Clean entire old runs if they are no longer needed:
        ```bash
        python main_runner.py --clean --run_id <very_old_run_id> --what all
        ```
3.  **Move `outputs/` Directory**: If the `outputs/` directory is on a partition with limited space, consider moving it to a larger disk and creating a symbolic link from the original location to the new one. Ensure your file system supports symlinks properly.
    ```bash
    # Example (Linux):
    # mv outputs /path/to/larger_disk/REPLAY_Influence_outputs
    # ln -s /path/to/larger_disk/REPLAY_Influence_outputs outputs
    ```

---

## 📚 **Further Reading**

- **[Influence Replay Algorithm Deep Dive](../technical/influence-replay-algorithm.md)** - Detailed step-by-step explanation of the influence replay algorithm and its implementation.
- **[Technical Deep Dive on REPLAY Algorithm & Determinism](../technical/comprehensive-analysis.md)**
- **[Main Project README](../../README.md)** (for up-to-date CLI commands and general project overview)
- **[Configuration File (`src/config.py`)]**(for model, training, and algorithm parameters)

---

## 🎯 **Summary**

Memory-efficient replay is a feature of the `MagicAnalyzer` class designed to handle large-scale influence analysis by storing intermediate training states (batch data, optimizer states) on disk rather than entirely in RAM.

- ✅ **Reduces RAM footprint** significantly for the collection of per-step replay data, making larger analyses feasible on memory-constrained systems.
- ✅ **Maintains mathematical correctness** by saving and loading all necessary information (images, labels, indices, LR, momentum buffers) for each step of the replay.
- ✅ **Enables analysis over longer training histories or larger datasets** that would be impossible with purely in-memory approaches.

**Important Developer Note**:
- The `MagicAnalyzer(use_memory_efficient_replay=Boolean)` parameter controls this behavior.
- Currently, `main_runner.py --magic` defaults to `use_memory_efficient_replay=False` (in-memory).
- To utilize memory-efficient replay via `main_runner.py`, a developer would need to modify its instantiation of `MagicAnalyzer` or add a specific CLI flag to control this parameter.

For detailed operational logs, especially when troubleshooting or verifying modes, use the `--log_level DEBUG` flag with `main_runner.py`. 