# Testing Guide for REPLAY Influence Analysis

This document provides comprehensive instructions for running and understanding the test suite for the REPLAY Influence Analysis project.

## Executive Summary

The REPLAY Influence Analysis project features a comprehensive test suite designed to ensure functionality, robustness, and consistency. The tests cover various aspects from individual components to integrated workflows.

## Test Structure

The test suite is primarily organized under the `tests/` directory:

```
tests/
├── __init__.py
├── helpers/
│   ├── __init__.py
│   └── test_helpers.py                  # Reusable helper functions for tests
├── integration/
│   ├── __init__.py
│   ├── test_component_consistency.py   # Integration tests for component determinism
│   ├── test_complete_workflows.py      # End-to-end workflow simulations
│   ├── test_data_pipeline_validation.py # Comprehensive data pipeline and determinism validation
│   └── test_run_management.py          # CLI and run management feature tests
# Note: Unit tests for individual modules might be co-located with src files or in a dedicated unit/ subfolder if added.
```

### Test Focus Areas

| Test Category         | Key Focus Areas                                      |
|-----------------------|------------------------------------------------------|
| **Integration Tests** | Component interaction, workflows, determinism, CLI   |
| **Helpers**           | Reusable test utilities, custom assertions         |

*(Specific unit tests for modules like `config.py`, `utils.py`, etc., would typically reside in `tests/unit/` if following a strict split, or be alongside the source files.)*

### Key Features of Test Structure

1. **Comprehensive Coverage**: Tests aim to cover core functionality, CLI interactions, deterministic behavior, and workflow validation.
2. **Professional Organization**: Tests are grouped by their scope (e.g., integration).
3. **Advanced Testing Patterns**: Includes deterministic testing, and simulations of various scenarios.
4. **CI/CD Ready**: Tests can be executed via `pytest` for automated checks.

## Test Categories and Detailed Coverage

### 1. Integration Tests (`tests/integration/`)
These tests validate the interaction between different components of the system and cover end-to-end scenarios.

- **`test_run_management.py`**: Focuses on the `main_runner.py` script, testing CLI argument parsing, run creation, cleanup, listing, and information display for various scenarios.
- **`test_component_consistency.py`**: Ensures that deterministic creation functions (e.g., for models, optimizers) behave consistently, producing identical components given identical seeds and instance IDs, and different components for different instance IDs.
- **`test_data_pipeline_validation.py`**: Critically examines the data loading pipeline, especially the custom `DeterministicSampler` and `seed_worker`, to ensure consistent data ordering across epochs and dataloader instances when intended.
- **`test_complete_workflows.py`**: Simulates higher-level workflows, such as the MAGIC-to-LDS pipeline (at a conceptual level, mocking heavy computations) and verifies deterministic reproducibility of model outputs.

### 2. Robustness & Error Handling
Implicitly tested across various integration tests, including:
- Handling of existing/missing run directories and artifacts.
- Correct parsing and rejection of invalid CLI argument combinations.
- Resilience to some forms of corrupted or missing dummy data in test setups.

### 3. Determinism
- Extensively tested in `test_component_consistency.py` and `test_data_pipeline_validation.py`.
- Ensures models, dataloaders, and optimizers are created deterministically.
- Verifies data shuffling and ordering consistency.

## Prerequisites

1. **Install test dependencies:**
   (Ensure `pytest` and any plugins like `pytest-cov` are in your environment. These are typically included in `requirements-test.txt` or a general `requirements.txt`.)
   ```bash
   pip install pytest pytest-cov pytest-mock
   # Or if you have a requirements-test.txt: 
   # pip install -r requirements-test.txt 
   ```

2. **Ensure the project environment is set up:**
   ```bash
   # Make sure you're in the project root directory
   cd /path/to/REPLAY_Influence

   # Activate your virtual environment if using one
   source venv/bin/activate  # or your preferred activation method
   ```

## Running Tests

### Basic Test Execution

**Run all tests found by pytest (from project root):**
```bash
pytest
```

**Run tests with verbose output:**
```bash
pytest -v
```

### Running Tests by Category/Path

**Integration tests only (if using markers and `pytest.ini` is configured):**
```bash
pytest -m integration
```
(Note: Ensure the `integration` marker is registered in `pytest.ini` to avoid warnings.)

**Run all tests in a specific directory:**
```bash
pytest tests/integration/
```

### Running Specific Test Files

**Run a specific test file:**
```bash
pytest tests/integration/test_run_management.py
pytest tests/integration/test_data_pipeline_validation.py
```

**Run a specific test function within a file:**
```bash
pytest tests/integration/test_component_consistency.py::test_model_initialization_consistency
```

### Advanced Test Options

**Run tests with coverage report (HTML):**
```bash
pytest --cov=src --cov-report=html
```
This will generate a coverage report in an `htmlcov/` directory.

**Run tests in parallel (if `pytest-xdist` is installed):**
```bash
pytest -n auto
```

**Show the 10 slowest tests:**
```bash
pytest --durations=10
```

## Test Quality Standards

The test suite aims for:
1. **Functionality Validation**: Ensuring core features work as expected.
2. **Determinism**: Verifying reproducible behavior of key components.
3. **Integration**: Testing interactions between different parts of the system.
4. **CLI Robustness**: Checking command-line interface behavior for various scenarios.

### Implementation Testing Focus

#### **1. Determinism & Consistency**
- Validating that models, data loaders, and optimizers initialize and behave identically given the same seeds and configurations.
- Ensuring data ordering is reproducible.

#### **2. Workflow & Run Management**
- End-to-end testing of CLI commands for MAGIC and LDS runs (simulated where necessary).
- Verification of run directory creation, artifact management, and cleanup processes.

#### **3. Core Logic (via Integration)**
- Testing the integration of `MagicAnalyzer` and `LDSValidator` concepts through `main_runner.py` actions.
- Validating data flow and handling of intermediate artifacts like checkpoints and scores in test scenarios.

## Continuous Integration Setup

### **Example CI Commands**
```bash
# Complete test suite for CI, with coverage and XML reports
pytest -v --cov=src --cov-report=xml:coverage.xml --junitxml=test_results.xml

# Optionally, run specific test groups if markers are used
# pytest -m integration
```

### **Test Markers**
The project currently uses the `integration` marker for integration tests. To avoid warnings, register it in `pytest.ini`:
```ini
# pytest.ini
[pytest]
markers =
    integration: marks tests as integration tests
    slow: marks tests as slow to run (example, if used)
```

### **Optional Dependencies**
Tests requiring optional dependencies should handle their absence gracefully, e.g., using `pytest.importorskip`.
```python
# Example in a test file:
# matplotlib = pytest.importorskip("matplotlib")
```

## Debugging Tests

### **Verbose Test Output**
For detailed test output, including print statements:
```bash
pytest -v -s
```

### **Debug Specific Test**
To debug a specific failing test with full traceback:
```bash
pytest tests/integration/test_run_management.py::test_scenario1_default_new_magic_run -vv --tb=long
```

### **Test Fixtures**
The primary fixture used across `test_run_management.py` is `cleanup_runs_before_after_each_test` (defined in the same file) which ensures a clean `outputs/runs` directory for each test. Other tests might use `tmp_path` provided by pytest for temporary file operations.

## Testing Best Practices

1. **Test Organization**: Group tests by scope (e.g., integration for `main_runner.py` interactions).
2. **Isolation**: Aim for independent tests, though integration tests inherently have more setup.
The `cleanup_runs_before_after_each_test` fixture helps isolate `test_run_management.py` scenarios.
3. **Coverage**: Strive for good coverage of critical paths and CLI options.

### **Maintenance and Updates**

1. **Regular Testing**: Run tests frequently, especially before and after code changes.
2. **Update Tests**: When adding or modifying features in `main_runner.py` or core components, update or add corresponding integration tests.
3. **Coverage Reporting**: Periodically check test coverage to identify untested areas.

This testing guide ensures the REPLAY Influence Analysis project maintains quality standards with comprehensive validation across all system components.