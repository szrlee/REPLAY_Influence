# REPLAY Influence Analysis Project

---

## 🎯 **Overview**

This project implements state-of-the-art methods for understanding how individual training data points affect a model's predictions and internal states. It provides a **research implementation** with comprehensive quality assurance and deterministic reproducibility.

### **Core Methodologies**

1. **🔮 MAGIC Influence Analysis**: Computes influence scores for training samples using advanced TracIn/REPLAY methodologies. These scores identify which training examples were most responsible for a model's behavior on specific test instances.

2. **🔬 LDS Validation System**: Empirically validates computed influence scores by training multiple models on systematically generated subsets of training data.

The system uses **CIFAR-10 dataset** and **ResNet9 architecture** with **complete deterministic reproducibility**.

---

## 🏆 **Key Features**

### 🔧 **Algorithmic Correctness**
- ✅ **6-Test Verification System**: Comprehensive data ordering verification
- ✅ **Research-Grade Reproducibility**: SHA256-based seed derivation
- ✅ **Configuration Validation**: Automatic parameter validation
- ✅ **Cross-Platform Determinism**: Identical results everywhere

### 🚀 **Performance & Memory**
- ✅ **Memory-Efficient Mode**: Step-specific loading (Note: `MagicAnalyzer` in `main_runner.py` currently hardcodes `use_memory_efficient_replay=False`)
- ✅ **Smart Caching**: Automatic detection of existing results
- ✅ **Professional Logging**: Progress tracking with time estimates

### 🛡️ **Production Quality**
- ✅ **100% Type Coverage**: Complete type annotations
- ✅ **Comprehensive Error Handling**: Exception management with detailed logging
- ✅ **Comprehensive Testing**: Integration tests covering core workflows.

---

## 📚 **Documentation**

**📖 [Complete Documentation Hub →](docs/README.md)**

| Guide Type | Purpose | Key Documents |
|------------|---------|---------------|
| **Quick Start** | Get running fast | This README |
| **Technical** | Implementation details | [Technical Analysis](docs/technical/comprehensive-analysis.md)<br>[Influence Replay Algorithm Deep Dive](docs/technical/influence-replay-algorithm.md) |
| **User Guides** | Practical usage | [Memory Efficient Mode](docs/guides/memory-efficient-replay.md) (Note: CLI flag for this is not currently exposed in `main_runner.py` for MAGIC) |
| **Quality** | Testing & validation | [Quality Report](docs/quality/comprehensive-report.md) |

---

## 🚀 **Quick Start**

### **1. Setup**
```bash
# Create environment and install dependencies
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Verify installation by showing configuration
python main_runner.py --show_config
```

### **2. Run Analysis**
```bash
# Run new MAGIC analysis (creates a new run directory)
python main_runner.py --magic

# Run MAGIC analysis, forcing retraining and recomputation even if artifacts exist
python main_runner.py --magic --force

# Run MAGIC analysis, specifying a custom run ID for the new run
python main_runner.py --magic --run_id my_magic_run_001

# Skip training for an existing MAGIC run and only recompute scores (requires --run_id)
python main_runner.py --magic --run_id existing_magic_run_id --skip_train --force

# Run LDS validation (requires an existing run_id that has MAGIC scores)
python main_runner.py --lds --run_id existing_magic_run_id

# Run LDS validation using an external scores file, creating a new run for LDS outputs
python main_runner.py --lds --run_id new_lds_run_id --scores_file path/to/external_scores.pkl

# Force LDS model retraining even if previous LDS results exist for the run
python main_runner.py --lds --run_id existing_magic_run_id --force
```

### **3. Run Management**
Each analysis run is stored in a timestamped directory (or a custom ID if specified) for easy tracking:
```bash
# List all runs
python main_runner.py --list

# Show detailed info about a specific run
python main_runner.py --info --run_id 20250129_143045_abc123
```

**Output Structure:**
```
outputs/
├── runs/
│   ├── 20250129_143045_abc123/  # Example timestamped/custom ID run
│   │   ├── checkpoints_magic/   # Model checkpoints (can be cleaned)
│   │   ├── checkpoints_lds/     # LDS model checkpoints (can be cleaned)
│   │   ├── scores_magic/        # Influence scores (preserved by default clean)
│   │   ├── losses_lds/          # LDS losses (preserved by default clean)
│   │   ├── plots_magic/         # Visualizations (preserved)
│   │   ├── plots_lds/           # LDS plots (preserved)
│   │   ├── logs_magic/          # MAGIC logs (always preserved)
│   │   ├── logs_lds/            # LDS logs (always preserved)
│   │   ├── magic_batch_dict.pkl # MAGIC batch data (preserved)
│   │   ├── indices_lds.pkl      # LDS subset indices (preserved)
│   │   └── run_metadata.json    # Configuration snapshot & run status
│   └── my_custom_run_001/       # Another example run
├── latest -> runs/YYYYMMDD_HHMMSS_xxxxxx  # Symlink to most recently created run
└── runs_registry.json           # Registry of all runs with metadata
```

### **4. Memory-Efficient Mode (Developer Note)**
The `MagicAnalyzer` class supports a `use_memory_efficient_replay` flag. However, `main_runner.py` currently instantiates `MagicAnalyzer` with this flag set to `False`. To enable memory-efficient MAGIC replay, this would need to be modified in `main_runner.py` or exposed as a CLI option.

📚 **[Memory Guide (Conceptual)→](docs/guides/memory-efficient-replay.md)**

---

## 📋 **Runner Options**

The `main_runner.py` script provides command-line options:

### **Core Actions (Mutually Exclusive)**
```bash
--magic                        # Run MAGIC influence analysis.
--lds                          # Run LDS validation. Requires --run_id.
--clean                        # Clean up run artifacts. Requires --run_id.
--list                         # List all available runs.
--info                         # Show detailed run information. Requires --run_id.
--show_config                  # Display current configuration and exit.
```

### **Common Arguments & Modifiers**
```bash
--run_id RUN_ID                # Specify a run ID.
                               # - For --magic (optional): Uses this ID for the new run.
                               # - For --lds: Specifies the run to use/create for LDS.
                               # - For --full_pipeline (optional): Uses this ID for MAGIC and then LDS.
                               # - For --clean, --info: Specifies the target run.
--force                        # Force recomputation/retraining.
                               # - For --magic: Forces model retraining and score recomputation.
                               # - For --lds: Forces retraining of LDS models.
--skip_train                   # For --magic: Skip training phase, use existing artifacts.
                               # Requires --run_id. Use with --force to recompute scores.
--scores_file SCORES_FILE      # For --lds: Path to an external MAGIC scores file.
                               # If provided, LDS outputs go into the --run_id directory.
--what [checkpoints|all]       # For --clean: What to clean.
                               # 'checkpoints' (default): Removes checkpoint dirs.
                               # 'all': Removes the entire run directory.
```

### **System Options**
```bash
--log_level LEVEL              # Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
                               # Default: INFO.
```

### **📝 Usage Examples**

```bash
# Run a new MAGIC analysis with default settings (new timestamped run_id created)
python main_runner.py --magic

# Run a new full pipeline (MAGIC then LDS)
python main_runner.py --full_pipeline

# Run MAGIC analysis and force retraining even if artifacts exist
python main_runner.py --magic --force

# Run a full pipeline and force all steps
python main_runner.py --full_pipeline --force

# Run MAGIC analysis with a specific run ID
python main_runner.py --magic --run_id my_first_magic_run

# Run a full pipeline with a specific run ID
python main_runner.py --full_pipeline --run_id my_pipeline_run

# For an existing MAGIC run 'my_first_magic_run', skip training and recompute scores
python main_runner.py --magic --run_id my_first_magic_run --skip_train --force

# Run LDS validation on the MAGIC results in 'my_first_magic_run'
# LDS outputs will be stored within 'my_first_magic_run'
python main_runner.py --lds --run_id my_first_magic_run

# Run LDS validation using an external scores file, storing LDS results in a new/existing run 'my_lds_run'
python main_runner.py --lds --run_id my_lds_run --scores_file path/to/scores.pkl

# Clean only checkpoint files from 'my_first_magic_run'
python main_runner.py --clean --run_id my_first_magic_run

# Clean all artifacts (delete entire directory) for 'my_first_magic_run'
python main_runner.py --clean --run_id my_first_magic_run --what all

# List all runs
python main_runner.py --list

# Get info for a specific run
python main_runner.py --info --run_id my_first_magic_run

# Show current configuration settings
python main_runner.py --show_config --log_level DEBUG
```

### **Automated Full Pipeline (MAGIC + LDS)**
`main_runner.py` now includes a `--full_pipeline` option to run both MAGIC and LDS sequentially.

```bash
# Run a new MAGIC analysis followed by LDS validation (new run_id generated by MAGIC)
python main_runner.py --full_pipeline

# Run a full pipeline, forcing all steps, using/creating a specific run_id
python main_runner.py --full_pipeline --run_id my_pipeline_run --force

# Run MAGIC (skipping training if run_id exists and is provided) then LDS
python main_runner.py --full_pipeline --run_id existing_magic_run_id --skip_train
```
When using `--full_pipeline`:
- The `--run_id` (optional) will be used for the MAGIC step and then passed to the LDS step.
- `--force` applies to both MAGIC and LDS stages.
- `--skip_train` (requires `--run_id`) applies to the MAGIC training phase.
- `--scores_file` is ignored, as the pipeline uses scores from its own MAGIC step.

---

## ⚙️ **Configuration**

Key settings in `src/config.py`:

```python
# Core Settings
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CIFAR_ROOT = "/tmp/cifar/" # Default path for CIFAR-10 data

# Training Parameters (used by both MAGIC and LDS model training)
MODEL_TRAIN_EPOCHS = 1 # Example: Quick test setting
MODEL_TRAIN_BATCH_SIZE = 1000
MODEL_TRAIN_LR = 0.025  # Peak LR for OneCycleLR

# Target Images
MAGIC_TARGET_VAL_IMAGE_IDX = 21
LDS_TARGET_VAL_IMAGE_IDX_FOR_CORRELATION = 21 # Should match MAGIC for direct validation

# LDS Specific
LDS_NUM_MODELS_TO_TRAIN = 2 # Example: Quick test setting
```
*Note: The example values for `MODEL_TRAIN_EPOCHS` and `LDS_NUM_MODELS_TO_TRAIN` in this README snippet are for brevity. Refer to `src/config.py` for actual defaults.*

---

## 🧹 **Cleanup & Management**

The `--clean` command helps manage disk space.

```bash
# Clean checkpoint files from a specific run (preserves scores, logs, plots)
python main_runner.py --clean --run_id your_run_id

# Clean ALL files from a specific run (deletes the run directory)
python main_runner.py --clean --run_id your_run_id --what all
```

**Note:**
- Default cleanup (`--what checkpoints`) removes only large checkpoint files to save disk space while preserving analysis results.
- `--what all` is destructive and will remove all data for the specified run.

---

## 🛠️ **Troubleshooting**

### **Common Issues**

| Issue | Solution |
|-------|----------|
| `CUDA out of memory` | If running MAGIC, this is currently hardcoded to non-memory-efficient. For LDS, ensure batch size is appropriate. Consider reducing `MODEL_TRAIN_BATCH_SIZE` in `src/config.py`. |
| `Data ordering mismatch` | Ensure `SEED` in `src/config.py` is consistent. Deterministic dataloaders are used. |
| `FileNotFoundError` for scores/checkpoints | Ensure the specified `run_id` is correct and contains the necessary artifacts. Use `--list` and `--info` to check. |
| `Configuration errors` | Run `python main_runner.py --show_config` to view current settings. Validate against `src/config.py`. |

### **Debug Mode**
Increase log verbosity for more detailed output:
```bash
python main_runner.py --magic --log_level DEBUG
```

📚 **[Conceptual Memory Guide →](docs/guides/memory-efficient-replay.md)**

---

## 📋 **Requirements**

### **System Requirements**
- **Python**: >=3.8
- **Memory**: 4GB+ RAM (More may be needed depending on batch sizes and model complexity).
- **Storage**: 2GB+ free space (more for multiple runs with full checkpoints).
- **Platform**: Linux, macOS, Windows (CUDA preferred for performance).

### **Core Dependencies**
(Refer to `requirements.txt` for exact versions)
```
torch
torchvision
numpy
matplotlib
tqdm
scipy
seaborn
```

---

## 📁 **Project Structure**

```
REPLAY_Influence/
├── src/                    # Core implementation
│   ├── magic_analyzer.py   # MAGIC influence calculation
│   ├── lds_validator.py    # LDS validation system
│   ├── model_def.py        # Model architecture definitions
│   ├── data_handling.py    # Dataset and DataLoader utilities
│   ├── config.py           # Configuration settings (constants, hyperparameters)
│   ├── run_manager.py      # Run directory and file path management
│   └── utils.py            # Utility functions, determinism
├── main_runner.py          # Main execution script
├── outputs/                # Generated results (runs, registry, latest link)
├── tests/                  # Unit and integration tests
│   ├── integration/
│   └── helpers/
├── docs/                   # Comprehensive documentation
└── requirements.txt        # Python dependencies
```

---

**🔗 [View Complete Documentation →](docs/README.md)**