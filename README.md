# dlbs
Repo for dlbs-minichallenge


## Repo Structure
```text
dlbs/
├─ configs/              # YAML configs for training/experiment runs (baseline, augmentation, sweeps, ...)
├─ dlbs/                 # Main Python package
│  ├─ plots/             # Plotting utilities (class distributions, image grids, validation viz)
│  ├─ summarizers/       # Dataset and object summary helpers
│  ├─ transform_data/    # Conversion of raw data into YOLO format
│  ├─ utils/             # Metrics and training helpers
│  │  ├─ metrics_standard.py            # Standard detection/segmentation metrics
│  │  ├─ metrics_stratified.py          # Metrics stratified by class/group
│  │  ├─ train_helpers.py               # Shared training utilities
│  │  └─ yolo_decode_validation_batch.py # Decode YOLO validation batches
│  ├─ wandb_api/         # Weights & Biases logging
│  │  ├─ custom_callback_yolo.py        # Custom YOLO training callback for W&B
│  │  └─ test_logging.py                # Log test-set metrics to W&B (used by train/test_model)
│  ├─ train.py           # Training entry point
│  └─ test_model.py      # Model evaluation entry point
├─ html_notebooks/       # Exploratory analysis and results/hypothesis html-notebooks
├─ notebooks/            # Exploratory analysis and results/hypothesis notebooks
├─ scripts/              # Shell scripts for data transformation and modeling runs (SLURM)
├─ setup.py
├─ LICENSE
├─ report_dlbs_juan_florian.pdf
├─ presentation.pptx
└─ README.md
```


## Setup Workspace

### Environment

```bash
# Create a new python environment
python3 -m venv .venv

# Activate it (Linux/macOS)
source .venv/bin/activate
```

### Install

Pick the extra that matches your hardware. The PyTorch CUDA build is pulled from the
matching index; only `cu126` and `cu128` are tested.

```bash
# FHNW SLURM cluster (CUDA 12.6)
pip install ".[calculon]" \
  --extra-index-url https://download.pytorch.org/whl/cu126

# Local GPU workstation (CUDA 12.8)
pip install ".[local]" \
  --extra-index-url https://download.pytorch.org/whl/cu128

# CPU / CI / Mac
pip install ".[cpu]" \
  --index-url https://download.pytorch.org/whl/cpu
```

###  Data Limits
Please provide the data in `data/` and adjust the data yaml paths if needed in the configs. The data is not included in the repo due to size and privacy, but the data transformation scripts are available in `scripts/`. 