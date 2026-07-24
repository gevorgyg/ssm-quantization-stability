# S4D Quantization Stability

Research code studying **quantization of the S4D state matrix Ā** — quantizing
the diagonal (complex) eigenvalues in polar / log-polar coordinates to prevent system instability, 
instead of the Cartesian grid used by Zhao et al. (arXiv:2506.12480), which can
round eigenvalues outside the unit circle and destabilize the recurrence.

Everything runs on sequential MNIST (sMNIST) with a small S4D model (`d_model=16, n_layers=4`).

## Requirements

- **Python ≥ 3.11**
- [**uv**](https://docs.astral.sh/uv/) for dependency management
  (`curl -LsSf https://astral.sh/uv/install.sh | sh`, or `brew install uv`)

## 1. Install the Python packages

```bash
uv sync
```

This creates a `.venv/` and installs the runtime dependencies. Prefix any command
with `uv run` to execute it inside that environment (e.g. `uv run python …`).

## 2. Train the S4D baseline

A pretrained checkpoint is **already included** at `s4/checkpoint/ckpt.pth` (≈98.7% val). To retrain from scratch:

```bash
cd s4
uv run python -m example --dataset mnist --d_model 16 --n_layers 4 \
    --weight_decay 0.0 --epochs 30 --num_workers 0
cd ..
```

- MNIST is downloaded automatically to `s4/data/` on first run.
- The best checkpoint is written to `s4/checkpoint/ckpt.pth` — the exact path the notebook loads.

## 3. Run the notebook

```bash
uv run jupyter lab
```

Open **`explore.ipynb`** and run the cells top to bottom. It loads `s4/checkpoint/ckpt.pth`,
downloads MNIST to `s4/data/` if needed, and runs the full study: the quantizers (Ā in
cartesian / polar / log-polar), the PTQ scheme comparison, streaming-divergence figure,
coordinate ablations, error-feedback state quantization, and the QAT arms.

## Project layout

| path | what it is |
|---|---|
| `explore.ipynb` | the main notebook — quantizers, model, all experiments |
| `s4/` | vendored S4/S4D reference implementation (used to train + load the baseline) |
| `s4/checkpoint/ckpt.pth` | pretrained full-precision S4D-16h baseline |
| `results/` | folder with experiment results |
| `NOTICE` | attribution for code adapted from the S4 repository |

## Attribution & license

This project is MIT-licensed (`LICENSE`). It reuses and adapts code from the S4 repository
(HazyResearch/state-spaces, Apache-2.0) — see `NOTICE` and `s4/LICENSE` for details.
