# ⚙️ Influence Replay Algorithm Tuning & Configuration

**Document Version**: 1.0.0
**Date**: Project Update

---

This document details specific configuration parameters available for tuning the behavior of the influence replay algorithm, particularly within the `MagicAnalyzer`. These settings allow users to manage trade-offs between computational fidelity, numerical stability, and resource usage.

## 1. Replay Clipping Mechanisms (MAGIC Analysis)

During the replay phase of the MAGIC analysis (`MagicAnalyzer._replay_single_step`), gradients and simulated parameters can sometimes grow very large, potentially leading to numerical instability (NaN/Inf values) and impacting the meaningfulness of the resulting influence scores. To mitigate this, configurable clipping mechanisms are provided.

These settings are located in `src/config.py` and control how and when clipping is applied during the replay loop.

### 1.1. Configuration Parameters

-   **`MAGIC_REPLAY_ENABLE_GRAD_CLIPPING`**
    -   **Type**: `bool`
    -   **Default**: `True`
    -   **Description**: Enables or disables gradient clipping for the gradients \\(g_t(\\mathbf{s}_t, \\mathbf{w})\\) computed during the replay. These are the gradients of the training loss (for batch \\(b_{t+1}\\)) with respect to the model parameters \\(\mathbf{s}_t\\) at replay step \\(t\\).
    -   **Impact**: If `True`, gradients exceeding `MAGIC_REPLAY_MAX_GRAD_NORM` will be scaled down.

-   **`MAGIC_REPLAY_MAX_GRAD_NORM`**
    -   **Type**: `float`
    -   **Default**: `0.5`
    -   **Description**: Sets the maximum permissible norm for the gradients \\(g_t\\) if `MAGIC_REPLAY_ENABLE_GRAD_CLIPPING` is `True`.
    -   **Impact**: Helps prevent excessively large gradients from destabilizing the simulation of the SGD step \\(h_t\\) or subsequent calculations involving \\(Q_t\\).

-   **`MAGIC_REPLAY_ENABLE_PARAM_CLIPPING`**
    -   **Type**: `bool`
    -   **Default**: `True`
    -   **Description**: Enables or disables parameter norm warnings and hard clipping for the simulated parameters \\(\mathbf{s}_{t+1}^{\text{sim}}\\) produced by `_simulate_sgd_step_for_replay`.
    -   **Impact**: If `True`, warnings are issued if parameter norms exceed `MAGIC_REPLAY_MAX_PARAM_NORM_WARNING`, and hard clipping is applied if they exceed `MAGIC_REPLAY_PARAM_CLIP_NORM_HARD`.

-   **`MAGIC_REPLAY_MAX_PARAM_NORM_WARNING`**
    -   **Type**: `float`
    -   **Default**: `5.0`
    -   **Description**: If `MAGIC_REPLAY_ENABLE_PARAM_CLIPPING` is `True`, this threshold triggers a warning log if the norm of any parameter tensor in \\(\mathbf{s}_{t+1}^{\text{sim}}\\) exceeds this value.
    -   **Impact**: Provides an indication of potentially problematic parameter scaling during replay.

-   **`MAGIC_REPLAY_PARAM_CLIP_NORM_HARD`**
    -   **Type**: `float`
    -   **Default**: `10.0`
    -   **Description**: If `MAGIC_REPLAY_ENABLE_PARAM_CLIPPING` is `True`, this threshold is used to hard-clip the norm of any parameter tensor in \\(\mathbf{s}_{t+1}^{\text{sim}}\\) that exceeds it. The parameter tensor is scaled down to have this norm.
    -   **Impact**: Acts as a stronger safeguard against extreme parameter values that could lead to NaNs or Infs in later calculations (e.g., when computing \\(Q_t = (\Delta_{t+1})^T \cdot \mathbf{s}_{t+1}^{\text{sim}}\\)).

### 1.2. Usage and Considerations

-   **Default Behavior**: By default (as per current `src/config.py`), clipping is active for both gradients and parameters. This prioritizes numerical stability, aiming to ensure that the replay computation completes successfully even if the underlying dynamics might be prone to explosion.
-   **Impact on Influence Scores**: When clipping is active, the computed influence scores (\\(\beta_t\\) and the propagated adjoints \\(\Delta_t\\)) are technically for a *modified* replay process where gradients or parameters were constrained. This is an approximation that balances fidelity with stability.
-   **Experimentation**: Users can disable these clipping mechanisms (by setting the `ENABLE` flags to `False` in `src/config.py`) or adjust the thresholds to observe the impact on influence scores.
    -   Disabling clipping might yield "purer" influence scores if the replay remains stable.
    -   However, it also increases the risk of numerical issues (NaN/Inf values), especially if the training or replay dynamics are inherently sensitive (e.g., due to learning rates, data characteristics, or the nature of the target task \\(\phi\\)).
-   **Debugging Instability**: If NaN/Inf values occur during replay, enabling and potentially tightening these clipping thresholds (e.g., lowering `MAX_GRAD_NORM` or `PARAM_CLIP_NORM_HARD`) can be a first step in diagnosing and managing the instability. Conversely, if scores seem overly dampened, cautiously relaxing clipping might be explored.

These configurable options provide a means to tailor the replay algorithm's behavior to specific experimental needs and datasets, balancing the desire for unconstrained replay against the practical requirement for numerical stability. 