# GPT from Scratch

A clean, modern reimplementation of a character-level GPT language model in PyTorch. This project is built for educational purposes but uses some modern practices like **GELU activations**, and **Cosine Learning Rate Scheduling**.

Current dataset: **Tiny Shakespeare**.

## Features

- **Modern Architecture**:
  - `nn.GELU` activations instead of `ReLU`.
  - Proper weight initialization (`std=0.02` with residual scaling) for stability.
- **Reproducible Environment**:
  - Uses [Pixi](https://pixi.sh) for lockfile-based dependency management.
- **Training Stability**:
  - Cosine Annealing Learning Rate Scheduler (`3e-4` max LR).

## Setup & Usage

This project uses **Pixi** to manage dependencies (Python 3.11, PyTorch 2.x).

1.  **Install Pixi** (if you haven't already):
    ```bash
    curl -fsSL https://pixi.sh/install.sh | bash
    ```

2.  **Run the Training**:
    The environment is automatically handled by Pixi. Just run:
    ```bash
    pixi run python main.py --comment "It is compulsory to provide a comment"
    ```

    This will:
    - Download dependencies (if needed).
    - Train the model for 5000 steps.
    - Save results to `results/run_<timestamp>/`.

## Codebase Overview

- **`main.py`**: The single-file implementation containing the `BigramLanguageModel`, training loop, and data handling.
- **`pixi.toml`**: Defines project dependencies and environment.
- **`data/`**: Contains the training dataset (`tinyshakespeare.txt`).

## Reproducibility

To ensure reproducibility, we set explicit seeds:
- PyTorch seed: `1337`
- The `pixi.lock` file ensures exact library versions are used across environments.