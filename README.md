# Streamflow Forecasting with Transfer Learning

Deep learning streamflow forecasting for **Lamar River** (Yellowstone NP) and **Hoh River** (Olympic NP), using NeuralHydrology with CAMELS pre-training and single-basin fine-tuning.

## Current Approach

**Transfer learning** following Kratzert et al. (2021):
1. Pre-train a CudaLSTM on 531 CAMELS US basins (Daymet forcing)
2. Fine-tune on target watershed with local GridMET climate data

| Watershed | NSE | KGE | Architecture |
|-----------|-----|-----|-------------|
| Lamar River | 0.89 | 0.94 | CudaLSTM (hidden=256) |
| Hoh River | 0.72 | 0.81 | CudaLSTM (hidden=256) |

## Quick Start

```bash
# 1. Setup environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Download data (see Data Acquisition below)

# 3. Fine-tune Lamar model
cd transfer_learningv2
nh-run finetune --config-file config/lamar_finetune_3var.yml

# 4. Evaluate
nh-run evaluate --run-dir runs/<run_name> --epoch 30 --period test

# 5. Dashboard
streamlit run dashboard.py
```

## Project Structure

```
.
├── transfer_learningv2/          # Main model code
│   ├── config/                   # Training configs (YAML)
│   │   ├── lamar_finetune_3var.yml    # Lamar fine-tuning
│   │   ├── hoh_finetune_3var.yml      # Hoh fine-tuning
│   │   ├── lamar_scratch_3var.yml     # Lamar from-scratch baseline
│   │   ├── hoh_scratch_3var.yml       # Hoh from-scratch baseline
│   │   └── camels_pretrain_531.yml    # CAMELS pre-training
│   ├── src/                      # Data prep pipeline (01-04)
│   ├── scripts/                  # Base model conversion
│   ├── tests/                    # Pipeline validation tests
│   ├── base_model_modified/      # Pre-trained CAMELS checkpoint
│   ├── dashboard.py              # Streamlit evaluation dashboard
│   ├── data/                     # NH-formatted NetCDF + attributes
│   └── runs/                     # Training run outputs
├── transfer_learning/            # V1 approach (kept for reference)
├── src/                          # Data acquisition & water balance
│   ├── download_*.py             # USGS, GridMET, SNOTEL downloaders
│   ├── water_balance.py          # Physics-based hydrological model
│   └── validate_*.py             # Validation scripts
├── data/                         # Raw data (streamflow, climate, SNOTEL)
├── runs/                         # CAMELS pre-training run
│   └── camels_pretrain_531_.../  # Base model source
├── archive/                      # Historical notebooks, models, docs
└── TRANSFER_LEARNING_GUIDE.md    # Scientific background
```

## Reproducing the Model

### Prerequisites
- Python 3.13 with `pip install -r requirements.txt`
- NVIDIA GPU with CUDA (tested on GTX 1070)
- NeuralHydrology (`nh-run` CLI)

### Step 1: Download Raw Data

```bash
source .venv/bin/activate

# Streamflow (USGS)
python src/download_lamar_river_data.py
python src/download_hoh_river_data.py

# Climate (GridMET elevation bands)
python src/download_gridmet_generalized.py

# SNOTEL snowpack
python src/download_snotel.py
```

### Step 2: Prepare NeuralHydrology Data

Run the numbered pipeline scripts from `transfer_learningv2/`:

```bash
cd transfer_learningv2

# Lump elevation bands to basin-mean climate
python src/01_prep_climate_lumped.py

# Convert streamflow CFS → mm/day
python src/02_convert_streamflow.py

# Merge climate + streamflow into NetCDF
python src/03_merge_to_netcdf.py

# Calculate static basin attributes
python src/04_calculate_attributes.py
```

### Step 3: Download CAMELS Dataset (for pre-training only)

The pre-trained base model is included in `transfer_learningv2/base_model_modified/`.
To re-train from scratch on CAMELS:

```bash
# Download CAMELS US dataset (~6GB)
# See: https://gdex.ucar.edu/dataset/camels/
# Place in transfer_learning/data/CAMELS_US/

# Pre-train
cd transfer_learningv2
nh-run train --config-file config/camels_pretrain_531.yml

# Convert base model for fine-tuning (renames CAMELS variable names to NetCDF-compatible)
python scripts/create_modified_base_model.py \
    --base-run-dir ../runs/<camels_run_name> \
    --output-dir base_model_modified
```

### Step 4: Fine-tune

```bash
cd transfer_learningv2

# Lamar River
nh-run finetune --config-file config/lamar_finetune_3var.yml

# Hoh River
nh-run finetune --config-file config/hoh_finetune_3var.yml
```

Training runs are saved to `transfer_learningv2/runs/<experiment_name_DDMM_HHMMSS>/`.

### Step 5: Evaluate

```bash
# Evaluate on test period (Oct 2020 - Dec 2024)
nh-run evaluate --run-dir runs/<run_name> --epoch 30 --period test

# Interactive dashboard
streamlit run dashboard.py
```

### Step 6: Run Tests

```bash
cd transfer_learningv2
python -m pytest tests/ -v
```

## Model Details

**Architecture:** CudaLSTM (Kratzert et al. 2021)
- Hidden size: 256
- Output dropout: 0.4
- Sequence length: 365 days
- Output activation: linear

**Inputs (3 dynamic variables):**
- `prcp_mm_day` — daily precipitation (mm)
- `tmax_C` — daily max temperature (°C)
- `tmin_C` — daily min temperature (°C)

**Static attributes (7):**
- `elev_mean`, `slope_mean`, `area_gages2`, `frac_forest`
- `soil_depth_pelletier`, `sand_frac`, `clay_frac`

**Fine-tuning:**
- Loss: MSE (smoother gradients than NSE for single-basin)
- Learning rate: 1e-4
- Epochs: 30
- Fine-tuned modules: LSTM + head

**Date splits:**
- Train: Oct 1991 – Sep 2015
- Validation: Oct 2015 – Sep 2020
- Test: Oct 2020 – Dec 2024

## Watersheds

- **Lamar River** — Snowmelt-dominated, high persistence (lag-1 autocorrelation = 0.986). SNOTEL SWE is critical.
- **Hoh River** — Rain-dominated, rapid storm response (lag-1 autocorrelation = 0.746). Harder to predict peaks.

## Historical Development

The project evolved through stages 1-6 using custom PyTorch LSTMs (see `archive/notebooks/`). The current NeuralHydrology transfer learning approach supersedes those experiments. See `TRANSFER_LEARNING_GUIDE.md` for scientific background.
