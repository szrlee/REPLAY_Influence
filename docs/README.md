# REPLAY Influence Analysis - Project Documentation

Welcome to the documentation for the REPLAY Influence Analysis project. This collection of documents provides in-depth information about the project's algorithms, implementation, usage, and quality assurance.

For a general project overview, installation instructions, and primary CLI usage examples, please refer to the **[Main Project README.md](../../README.md)**.

## 📚 Documentation Sections

This documentation is organized into the following main sections:

1.  **[Technical Deep Dives (`technical/`)](technical/)**: Contains detailed explanations of the core algorithms, architectural choices, and specific technical implementations.
    *   **[Overall Technical Analysis (`comprehensive-analysis.md`)](technical/comprehensive-analysis.md)**: A central document summarizing key design aspects, including links to detailed algorithm descriptions, determinism strategies, and configuration options.
    *   **[Influence Replay Algorithm (`influence-replay-algorithm.md`)](technical/influence-replay-algorithm.md)**: A step-by-step breakdown of the influence replay algorithm.
    *   **[Determinism Strategy (`determinism-strategy.md`)](technical/determinism-strategy.md)**: Detailed explanation of how reproducibility is achieved.
    *   **[Replay Algorithm Tuning (`replay-algorithm-tuning.md`)](technical/replay-algorithm-tuning.md)**: Configuration options for the replay process, like clipping.
    *   **[ResNet-9 Implementation (`resnet9-implementation.md`)](technical/resnet9-implementation.md)**: Details of the ResNet-9 model variants used.

2.  **[User & Developer Guides (`guides/`)](guides/)**: Practical guides for using specific features or understanding certain aspects of the system from a user or developer perspective.
    *   **[Memory-Efficient Replay (`memory-efficient-replay.md`)](guides/memory-efficient-replay.md)**: Guide to understanding and using the memory-efficient replay mode in `MagicAnalyzer`.

3.  **[Quality Assurance (`quality/`)](quality/)**: Information regarding the testing, validation, and overall quality of the project.
    *   **[Quality Assurance Overview (`quality-assurance-overview.md`)](quality/quality-assurance-overview.md)**: An overview of the testing philosophy, types of tests, and quality metrics.
    *   **[Test Execution Guide (`test-execution-guide.md`)](quality/test-execution-guide.md)**: How to run the various tests.

## 🧭 Navigating the Documentation

To help you find the information you need, here are some suggested starting points based on your interest:

*   **To understand the core influence replay method:**
    *   Start with **[Influence Replay Algorithm Deep Dive](technical/influence-replay-algorithm.md)**.
    *   Then, review the **[Overall Technical Analysis](technical/comprehensive-analysis.md)** for context.

*   **To understand how reproducibility is achieved:**
    *   Read the **[Determinism Strategy](technical/determinism-strategy.md)**.

*   **To learn about using memory-efficient replay or configuring replay options:**
    *   See the **[Memory-Efficient Replay Guide](guides/memory-efficient-replay.md)**.
    *   Consult the **[Replay Algorithm Tuning](technical/replay-algorithm-tuning.md)** document.

*   **To understand the model architecture:**
    *   Refer to the **[ResNet-9 Implementation Guide](technical/resnet9-implementation.md)**.

*   **To learn about testing and quality assurance:**
    *   Begin with the **[Quality Assurance Overview](quality/quality-assurance-overview.md)**.
    *   Follow up with the **[Test Execution Guide](quality/test-execution-guide.md)**.

*   **For a high-level technical overview of the entire system:**
    *   The **[Overall Technical Analysis](technical/comprehensive-analysis.md)** is the best starting point.

## 🛠️ Key System Components (Brief Overview)

-   **Core Algorithm Logic**: `src/magic_analyzer.py` (REPLAY), `src/lds_validator.py` (LDS).
-   **Model Definitions**: `src/model_def.py`.
-   **Configuration**: `src/config.py`.
-   **Utilities & Determinism**: `src/utils.py`.
-   **Run Management**: `src/run_manager.py`.

This documentation aims to be a comprehensive resource. If you find areas that are unclear or could be improved, please consider contributing.

---
*This documentation suite is actively maintained. Last significant structural review: May 2025.*