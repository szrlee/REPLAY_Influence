# Test Execution Guide

**Document Version**: 1.0.0
**Date**: Project Update

---

This document provides comprehensive, practical instructions for setting up the environment and running the various test suites for the REPLAY Influence Analysis project.

For an overview of the project's Quality Assurance philosophy, testing strategy, and what aspects are covered by tests, please refer to the **[Quality Assurance Overview](quality-assurance-overview.md)**.

## 1. Prerequisites

Before running tests, ensure the following prerequisites are met:

1.  **Project Setup**: The project should be cloned, and you should be in the project root directory.
    ```bash
    cd /path/to/REPLAY_Influence
    ```

2.  **Python Environment**: A Python environment (version >=3.8) should be active. If using a virtual environment (recommended):
    ```bash
    # Example: Activate a venv
    source venv/bin/activate 
    ```

3.  **Test Dependencies**: All necessary testing libraries must be installed. These are typically listed in `requirements-test.txt` or a general `requirements.txt`.
    ```bash
    # Install pytest and common plugins
    pip install pytest pytest-cov pytest-mock pytest-xdist
    
    # Or, if you have a specific test requirements file:
    # pip install -r requirements-test.txt
    ```

## 2. Test Suite Structure Overview

The test suite is located under the `tests/` directory:
```
tests/
├── __init__.py
├── helpers/                # Reusable helper functions and fixtures for tests
│   ├── __init__.py
│   └── test_helpers.py     # (Example, actual structure may vary)
├── integration/            # Integration tests covering component interactions
│   ├── __init__.py
│   ├── test_component_consistency.py
│   ├── test_complete_workflows.py
│   ├── test_data_pipeline_validation.py
│   └── test_run_management.py
# Unit tests, if added, might reside in tests/unit/ or alongside source files.
```
-   **`tests/integration/`**: Contains tests that validate the interaction between different components, CLI operations, and end-to-end workflows.
-   **`tests/helpers/`**: Provides utility functions, custom assertions, and fixtures used by the tests.

## 3. Running Tests with Pytest

All tests are designed to be run using `pytest` from the project root directory.

### 3.1. Basic Test Execution

-   **Run all discovered tests:**
    ```bash
    pytest
    ```

-   **Run tests with verbose output (shows test names and status):**
    ```bash
    pytest -v
    ```

-   **Run tests with full verbose output and print statements (`-s` captures stdout):**
    ```bash
    pytest -v -s
    ```

### 3.2. Running Specific Tests

-   **Run all tests in a specific directory:**
    ```bash
    pytest tests/integration/
    ```

-   **Run a specific test file:**
    ```bash
    pytest tests/integration/test_run_management.py
    ```

-   **Run a specific test class within a file (if tests are class-based):**
    ```bash
    pytest tests/integration/test_some_file.py::TestSomeClass
    ```

-   **Run a specific test function (or method within a class):**
    ```bash
    pytest tests/integration/test_component_consistency.py::test_model_initialization_consistency
    ```

### 3.3. Using Test Markers

If tests are marked (e.g., `@pytest.mark.integration`), you can run specific groups.

-   **Run tests with a specific marker (e.g., `integration`):**
    ```bash
    pytest -m integration
    ```
    *Note: Markers must be registered in `pytest.ini` to avoid warnings. Example `pytest.ini` content:*
    ```ini
    # pytest.ini
    [pytest]
    markers =
        integration: marks tests as integration tests
        slow: marks tests as slow to run (example, if used)
    ```

### 3.4. Advanced Test Options

-   **Test Coverage Report (HTML)**:
    Requires `pytest-cov` plugin.
    ```bash
    pytest --cov=src --cov-report=html
    ```
    This command runs tests, collects coverage data for the `src/` directory, and generates an HTML report in the `htmlcov/` directory. Open `htmlcov/index.html` to view the report.

-   **Test Coverage Report (XML - for CI systems)**:
    ```bash
    pytest --cov=src --cov-report=xml:coverage.xml
    ```

-   **Run Tests in Parallel (Distribute tests across multiple CPUs)**:
    Requires `pytest-xdist` plugin.
    ```bash
    # Automatically use available CPUs
    pytest -n auto
    
    # Specify number of workers (e.g., 4)
    pytest -n 4
    ```

-   **Show the N Slowest Tests (e.g., top 10):**
    ```bash
    pytest --durations=10
    ```

-   **Stop on First Failure (`-x`):**
    ```bash
    pytest -x
    ```

-   **Run Only Failed Tests (`--lf` for "last failed"):**
    ```bash
    pytest --lf
    ```

## 4. Continuous Integration (CI) Setup

A typical CI pipeline might execute the following commands:

```bash
# Ensure all dependencies are installed (including test dependencies)
# pip install -r requirements.txt 
# pip install -r requirements-test.txt # If separate

# Run all tests with verbose output, generate XML coverage and JUnit XML test results
pytest -v --cov=src --cov-report=xml:coverage.xml --junitxml=test_results.xml
```
These XML files can then be processed by CI platforms to display test results and coverage information.

## 5. Debugging Tests

-   **Verbose Output**: Use `pytest -v` or `pytest -vv` for more detailed output about which tests are running and their status.
-   **Print Statements**: To see `print()` statements from your tests or code, run pytest with the `-s` option:
    ```bash
    pytest -s tests/integration/your_test_file.py
    ```
-   **Full Tracebacks**: For detailed error information on failures:
    ```bash
    pytest --tb=long your_failing_test_file.py
    ```
-   **Debugging with PDB (Python Debugger)**:
    Insert `import pdb; pdb.set_trace()` in your test code where you want to start debugging. Then run the specific test. Pytest will drop you into the PDB console.
    Alternatively, run pytest with `--pdb` to automatically enter PDB on failures.
    ```bash
    pytest --pdb your_failing_test_file.py
    ```

## 6. Test Fixtures

Pytest fixtures are used to set up and tear down resources needed by tests. Key fixtures include:

-   **`tmp_path` (built-in pytest fixture)**: Provides a temporary directory unique to each test function, useful for tests that need to read/write files without affecting the project directory.
-   **Custom Fixtures (e.g., in `tests/helpers/` or conftest.py)**: The project may define custom fixtures. For example, `test_run_management.py` uses a fixture like `cleanup_runs_before_after_each_test` (defined in the same file or a shared `conftest.py`) to ensure a clean `outputs/runs` directory for each test scenario involving run creation.

## 7. Best Practices for Writing and Maintaining Tests

(This section is more for developers contributing tests)

-   **Isolation**: Aim for tests that are independent of each other. Setup and teardown fixtures help achieve this.
-   **Clarity**: Write tests that are easy to understand. Use descriptive names for test functions and variables.
-   **Coverage**: Strive for good coverage of critical code paths, edge cases, and common user scenarios.
-   **Maintenance**: When code in `src/` changes, corresponding tests should be updated or new tests added to reflect these changes.
-   **Regular Execution**: Run tests frequently during development to catch regressions early.

This guide provides the necessary information to effectively run and manage the test suite for the REPLAY Influence Analysis project. 