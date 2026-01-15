# LLM Lab

A modular, ground-up implementation of GPT designed for **Architectural Research** and **Optimization Dynamics** study.

Unlike standard "tutorial" repositories, this codebase is structured to facilitate rapid experimentation with new components (e.g., Attention variants, Norms) while providing deep diagnostic visibility into the training process.

## 🔬 Research-Grade Diagnostics
This repository implements advanced metrics to debug training stability beyond just "Loss":

- **Update-to-Data (UD) Ratio**: $\frac{\sigma_{\text{update}}}{\sigma_{\text{param}}}$. The "heartbeat" of training. We aim for $\sim 10^{-3}$. If too high, training is unstable; too low, it's inefficient.
- **Gradient Variance**: $\sigma^2_{\text{grad}}$. distinguishing between "hard samples" and "noisy optimization".
- **Dead Activation %**: Monitors the health of ReLU/GELU units to detect network capacity collapse.

## 📂 Project Structure

The codebase is refactored from a single script into a modular research framework:

- **`config.py`**: Centralized, type-safe configuration using `GPTConfig` dataclasses.
- **`model.py`**: Pure PyTorch implementation of the architecture. Designed for component swappability (e.g., replacing `MultiHeadAttention` with `LinearAttention`).
- **`train.py`**: The research training loop. Decoupled from the model definition, featuring heavy instrumentation for the diagnostics above.
- **`utils.py`**: Efficient CSV/JSONL logging for long-running experiments.

## 🚀 Workflow

### 1. Configure
Modify `config.py` or pass CLI arguments to define your experiment.
```bash
# Example: Probing the effect of strictly lower learning rates on dead activations
pixi run python train.py --learning_rate 1e-4 --activation relu
```

### 2. Train & Log
Metrics are streamed to `experiment_metrics.csv` every step.
```csv
step, tokens_sec, grad_norm, update_ratio, dead_perc, train_loss
0,    450.2,      10.5,      0.0012,       0.51,      4.65
...
```

### 3. Analyze
Use the CSV logs to generate insight plots. (See `notebooks/` - *coming soon*).

## Reproducibility
Managed via **Pixi**.
```bash
pixi run python train.py
```
Global seeds are set for Model initialization and Data sampling to ensure bit-exact reproducibility across runs.