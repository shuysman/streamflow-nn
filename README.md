# Neural Network Streamflow Forecasting

**A Physics-Informed, Probabilistic Deep Learning System for Hydrological Forecasting**

## 1. Overview

This project implements a sophisticated **Sequence-to-Sequence (Seq2Seq) Encoder-Decoder** network for forecasting 7-day streamflow on the Lamar River (Yellowstone National Park) and Hoh River (Olympic National Park).

Moving beyond naive deep learning baselines, the final system integrates **SNOTEL snowpack data**, **GridMET climate forcing**, and **Quantile Regression** to provide robust, risk-aware forecasts. It addresses the "Storage Problem" in hydrology by explicitly modeling snow water equivalent (SWE) and melt potential.

### Key Capabilities
*   **Probabilistic Forecasting:** Outputs 10th, 50th, and 90th percentiles (P10, P50, P90) to quantify uncertainty.
*   **Physics-Informed:** Incorporates hydrological principles like rain-on-snow events and elevation-banded climate forcing.
*   **Flood Risk Detection:** The P90 upper bound successfully captures extreme flood risks that median forecasts often miss.
*   **Calibration:** Achieves ~78% coverage for an 80% confidence interval.

## 2. Model Architecture (Stage 6)

The current state-of-the-art model in this repository represents "Stage 6" of development:

*   **Encoder:** A 2-layer LSTM that processes a 60-day lookback window.
    *   *Inputs:* Log-scaled Streamflow, SNOTEL SWE, Precipitation, Temperature, and Physics Features (e.g., Melt Potential).
*   **Decoder:** An Autoregressive LSTM forecasting a 7-day horizon.
    *   *Mechanism:* Uses its own P50 prediction to step forward in time (recurrent feedback).
*   **Loss Function:** Pinball Loss (Quantile Loss) for asymmetric penalty optimization.
*   **Training Strategy:** 3-Phase Robust Training (Teacher Forcing decay $\rightarrow$ Noise Injection) to prevent exposure bias and autoregressive explosion.

## 3. Performance

| Metric | Value | Description |
| :--- | :--- | :--- |
| **NSE (Median)** | ~0.94 | Nash-Sutcliffe Efficiency (Excellent) |
| **Coverage** | 78.2% | Percentage of observations falling between P10 and P90 (Target >75%) |
| **Peak Capture** | High | P90 captures ~87% of extreme peaks |

## 4. Setup & Installation

### Prerequisites
*   Python 3.13 (Primary) or 3.12 (Legacy)
*   NVIDIA GPU (Recommended for training) with CUDA support

### Environment
The project manages dependencies via `venv` and `pip`.

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Key Dependencies
*   `torch` (Deep Learning)
*   `pandas`, `numpy` (Data Manipulation)
*   `matplotlib`, `seaborn` (Visualization)
*   `jupyterlab` (Interactive Development)

## 5. Usage

### Data Acquisition
The project includes scripts to fetch live USGS streamflow and SNOTEL data.

```bash
# Download Lamar River Streamflow
python src/download_lamar_river_data.py

# Download SNOTEL Snowpack Data
python src/download_snotel.py
```

### Running Experiments (Notebooks)
The core model development is documented in progressive Jupyter notebooks located in `notebooks/`.

*   **Stage 6 (Final Model):** `notebooks/lamar_river_stage6_probabilistic_daily.ipynb`
*   **Hoh River Analysis:** `notebooks/hoh_river_stage6_35years.ipynb`

To launch:
```bash
jupyter lab
```

### Project Structure

```text
.
├── data/                   # Raw and processed datasets (Streamflow, Climate, SNOTEL)
├── notebooks/              # Jupyter notebooks for Stages 1-6
├── src/                    # Source code for data fetching and modeling
│   ├── water_balance.py    # Physics-based hydrological model
│   ├── download_*.py       # Data acquisition scripts
│   └── ...
├── output/                 # Model artifacts, plots, and logs
└── ...
```

## 6. Development History

The project evolved through distinct stages to solve specific hydrological modeling challenges:

*   **Stage 1 (Naive LSTM):** Baseline model. Failed to capture seasonal snowmelt.
*   **Stage 2 (Climate Forcing):** Added Precip/Temp. Improved seasonality but missed "Rain-on-Snow" events.
*   **Stage 3 (SNOTEL Integration):** Added Snow Water Equivalent (SWE). Massive accuracy jump (NSE ~0.94).
*   **Stage 4 (Seq2Seq):** Moved to Encoder-Decoder. Suffered from "exposure bias" (hallucinated floods).
*   **Stage 5 (Robustness):** Fixed stability issues with noise injection and scheduled sampling.
*   **Stage 6 (Probabilistic):** Final Quantile Regression model. Adds uncertainty bounds for risk management.

## 7. Documentation

For detailed developer guidelines, coding standards, and specific command references, please see:
*   `PROJECT_SUMMARY.md`: In-depth narrative of the model's evolution.
