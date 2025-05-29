# REPLAY Influence Analysis - Documentation

Welcome to the documentation for the REPLAY Influence Analysis project.

## 📚 Documentation Structure

### 🚀 [Quick Start Guides](guides/)
- **[Memory-Efficient Replay Guide](guides/memory-efficient-replay.md)** - Memory optimization strategies (Conceptual: CLI flag not directly exposed for MAGIC in `main_runner.py`)

### 🔧 [Technical Documentation](technical/)
- **[REPLAY Algorithm Analysis](technical/comprehensive-analysis.md)** - Complete technical analysis with seed management
- **[ResNet-9 Implementation Guide](technical/resnet9-implementation.md)** - Complete ResNet-9 documentation

### 🏆 [Quality Assurance](quality/)
- **[Comprehensive Quality Report](quality/comprehensive-report.md)** - System testing status
- **[Testing Guide](quality/testing-guide.md)** - Test execution instructions

## 🎯 **Quick Start**

### Basic Model Usage

```python
from src.model_def import construct_resnet9_paper
from src.utils import create_deterministic_model

# Create paper-compliant ResNet-9 model
model = construct_resnet9_paper(num_classes=10)

# Or create deterministically for reproducible experiments
model = create_deterministic_model(
    master_seed=42,
    creator_func=construct_resnet9_paper,
    instance_id="my_experiment"
)
```

### Running Analysis
(Refer to the main project README.md for detailed CLI examples)
```bash
# Run MAGIC analysis
python main_runner.py --magic

# Run LDS validation (requires a run_id with MAGIC scores)
python main_runner.py --lds --run_id your_magic_run_id
```

## 🎯 **Key Features**

### ✅ **ResNet-9 Paper Implementation**
- Paper-compliant 14.2M parameter model (Note: actual params depend on `MODEL_CREATOR_FUNCTION`)
- Proper parameter grouping with bias scaling
- LogSumExp pooling implementation

### ✅ **Research-Grade Determinism**
- Component-specific seed derivation using SHA256
- Perfect MAGIC/LDS consistency verification
- Comprehensive DataLoader worker handling

### ✅ **Production Quality**
- Comprehensive test suite covering core functionality
- Complete type annotations
- Memory-efficient replay mode (developer-configurable in `MagicAnalyzer`)
- Robust error handling and detailed logging

## 📁 **Source Code Organization**

- **Core Algorithm** (`src/magic_analyzer.py`) - Main REPLAY implementation
- **Validation** (`src/lds_validator.py`) - LDS validator for verification
- **Models** (`src/model_def.py`) - ResNet-9 implementations
- **Configuration** (`src/config.py`) - Centralized settings (constants, hyperparameters)
- **Run Management** (`src/run_manager.py`) - Handles creation and tracking of experiment runs and paths
- **Utilities** (`src/utils.py`) - Deterministic training utilities, helper functions

## 📋 **Documentation Reference**

- **[REPLAY Algorithm Analysis](technical/comprehensive-analysis.md)** - Complete technical details and algorithm fixes
- **[ResNet-9 Implementation Guide](technical/resnet9-implementation.md)** - Paper-compliant architecture details
- **[Memory-Efficient Replay Guide](guides/memory-efficient-replay.md)** - Performance optimization strategies
- **[Quality Report](quality/comprehensive-report.md)** - System status and test results
- **[Testing Guide](quality/testing-guide.md)** - Test execution and validation procedures

## ⚙️ Configuration

Key settings for hyperparameters, seeds, and model choices are primarily in `src/config.py`.
Run-specific directory and file path generation is handled by `src/run_manager.py`.

## 🎯 Key Features Documented

### ✅ **Implementation Features**
- **Comprehensive test suite** covering functionality and robustness
- **Error handling** with validation and logging
- **CLI interface** with workflow orchestration via `main_runner.py`

### ✅ **Security & Testing Features**
- Path traversal prevention and input sanitization considerations
- Numerical stability testing and NaN/Inf handling

### ✅ **Performance Features**
- Memory-efficient replay mode (developer-configurable)
- Deterministic operations for reproducible results
- Resource utilization monitoring and cleanup validation

## 📖 Quick Navigation

### For New Users
1. Start with **[ResNet-9 Implementation Guide](technical/resnet9-implementation.md)** for model architecture details
2. Review **[Testing Guide](quality/testing-guide.md)** for validation information
3. Check **[Comprehensive Quality Report](quality/comprehensive-report.md)** for testing details

### For Developers
1. **[REPLAY Algorithm Analysis](technical/comprehensive-analysis.md)** - Technical implementation details
2. **[Testing Guide](quality/testing-guide.md)** - Test suite organization and execution

### For Usage
1. **[Comprehensive Quality Report](quality/comprehensive-report.md)** - Testing and validation information
2. **[Memory-Efficient Replay Guide](guides/memory-efficient-replay.md)** - Performance optimization
3. **[Testing Guide](quality/testing-guide.md)** - Test execution instructions

## 🏗️ System Architecture Overview

The REPLAY Influence Analysis system consists of several key components:

### **Core Components**
- **MAGIC Analyzer** (`src/magic_analyzer.py`) - Influence function computation with replay
- **LDS Validator** (`src/lds_validator.py`) - Linear Datamodeling Score validation system
- **Model Definitions** (`src/model_def.py`) - ResNet-9 and other model implementations
- **Configuration Management** (`src/config.py`) - Centralized settings (constants, hyperparameters)
- **Run Management** (`src/run_manager.py`) - Handles creation and tracking of experiment runs and paths
- **Utilities** (`src/utils.py`) - Deterministic operations and helper functions

### **Quality Assurance**
- **Integration tests** covering core workflows and component interactions
- **Numerical stability** checks and NaN/Inf handling
- **Workflow validation** including error recovery in `main_runner.py`

### **Documentation Standards**
- **Technical specifications** for implementations
- **Usage examples** (see main project README.md for CLI)
- **Troubleshooting guides**

## 🎯 Usage Examples

### Basic ResNet-9 Model Usage
```python
from src.model_def import construct_resnet9_paper # Example model
from src.utils import create_deterministic_model
from src import config # To access config.NUM_CLASSES

# Create a model (e.g., paper-compliant ResNet-9)
model = construct_resnet9_paper(num_classes=config.NUM_CLASSES)

# Create deterministically for reproducible experiments
deterministic_model = create_deterministic_model(
    master_seed=config.SEED,
    creator_func=construct_resnet9_paper, # or any other model creator from model_def.py
    instance_id="my_experiment_model",
    num_classes=config.NUM_CLASSES # Ensure num_classes is passed if required by creator_func
)
```

### Complete Analysis Workflow (via CLI)
(Refer to the main project README.md for detailed and up-to-date CLI examples)
```bash
# Run complete MAGIC analysis (creates a new run)
python main_runner.py --magic

# Run LDS validation on an existing MAGIC run
python main_runner.py --lds --run_id <your_magic_run_id>

# View configuration
python main_runner.py --show_config
```

### Running Tests
```bash
# Run all tests (ensure pytest and plugins like pytest-cov are installed)
pytest

# Example: Run integration tests with verbose output
pytest tests/integration -v

# Generate coverage report (example)
pytest --cov=src --cov-report=html
```

## 📊 System Information

### **Test Coverage**
- **Integration Tests**: Validate component interactions and core workflows.
- (Refer to `pytest.ini` and test execution reports for detailed coverage metrics).

### **Testing Categories**
- **Functionality testing** for core algorithms (MAGIC, LDS).
- **Determinism validation** for reproducible outputs.
- **Configuration validation** and error handling.
- **Integration testing** for component interactions and run management.

## 🔄 Test Execution

### **Running Tests**
```bash
# Complete test suite (from project root)
pytest

# Generate HTML coverage report
pytest --cov=src --cov-report=html

# Run tests for a specific file
pytest tests/integration/test_run_management.py -v
```

### **Test Organization**
- Automated testing for algorithm correctness and workflow validation.
- Determinism checks for models, dataloaders, optimizers.
- Run management CLI interactions.

## 🎓 Learning Resources

### **For Research**
- **[REPLAY Algorithm Analysis](technical/comprehensive-analysis.md)** - Technical details and algorithm fixes
- **[ResNet-9 Implementation Guide](technical/resnet9-implementation.md)** - Architecture details

### **For Development**
- **[Testing Guide](quality/testing-guide.md)** - Testing standards and execution
- **[Quality Report](quality/comprehensive-report.md)** - Code quality information

### **For Implementation**
- **Testing and validation procedures** for safe operation
- **Performance optimization guidelines** (conceptual, e.g., memory efficiency)
- **Configuration management** for reproducible experiments (`src/config.py`)

## 🤝 Contributing

To contribute to the documentation:

1. **Follow the established structure** in the appropriate directory (`docs/guides`, `docs/technical`, `docs/quality`).
2. **Use consistent formatting** (Markdown) with existing documentation.
3. **Include code examples** where relevant, ensuring they are up-to-date with `main_runner.py` and `src/` APIs.
4. **Test all examples** to ensure they work correctly.
5. **Update this README** (`docs/README.md`) or the main project `README.md` when adding new sections or making significant changes to CLI/usage.

## 📞 Support

For questions about the documentation:
- Check the relevant section first (technical, quality, guides).
- Review the **[Testing Guide](quality/testing-guide.md)** for validation procedures.
- Consult the **[Comprehensive Quality Report](quality/comprehensive-report.md)** for production guidelines.

---

**Documentation Status**: 🚧 **Under Review & Update**
**Last Updated**: May 2025
**Coverage**: Core system components and workflows.