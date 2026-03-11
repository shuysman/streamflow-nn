# General Guide: Hydrological Transfer Learning with CAMELS

This guide outlines the standard workflow for applying Transfer Learning in hydrology, specifically adapting a model pre-trained on the large-scale CAMELS dataset (using Daymet forcing) to a specific target basin using diverse local data sources (e.g., GridMET).

## 1. Scientific Rationale: Handling Climate Bias
A common question in transfer learning is whether one can pre-train on one climate dataset (e.g., Daymet) and fine-tune on another (e.g., GridMET).

**The Answer: Yes.**
In this workflow, the "Fine-Tuning" phase serves a dual purpose:
1.  **Hydrological Adaptation:** The model learns the specific physical characteristics of the target watershed (snowpack dynamics, soil retention, topology).
2.  **Domain Adaptation (Bias Correction):** The model weights implicitly adjust to the statistical biases between the source forcing (Daymet) and the target forcing (GridMET). The model effectively "learns" the systematic differences (e.g., $GridMET_{temp} \approx Daymet_{temp} + \delta$) during the adaptation epochs.

This approach avoids the complex and error-prone need to manually bias-correct terabytes of pre-training data.

---

## 2. Prerequisites

### Software Environment
Ensure the [NeuralHydrology](https://neuralhydrology.github.io/) package is installed.
```bash
pip install neuralhydrology
```

### Source Data: CAMELS US
The base model requires the CAMELS dataset to learn general hydrological physics.
1.  **Register/Login:** [UCAR CAMELS Dataset](https://ral.ucar.edu/solutions/products/camels).
2.  **Download:**
    *   `camels_attributes_v2.0.zip`
    *   `basin_mean_forcing_daymet_v1.00.zip` (Standard Daymet forcing)
    *   `usgs_streamflow_v1.2.zip`
3.  **Structure:** Extract to a directory (e.g., `data/CAMELS_US/`) following this structure:
    ```text
    data/CAMELS_US/
    ├── basin_mean_forcing_daymet_v1.00/
    ├── camels_attributes_v2.0/
    └── usgs_streamflow_v1.2/
    ```

---

## 3. Workflow Steps

### Step 1: Target Data Preparation
Your local target data (e.g., for the Lamar River) must be formatted to match the inputs used by the base model.

1.  **Acquire Data:** Download your forcing data (e.g., GridMET).
2.  **Format Requirements:**
    *   **Format:** NetCDF (`.nc`) is the standard for generic NeuralHydrology datasets.
    *   **Variables:** Must match the base model training inputs. Typically:
        *   `prcp` (Precipitation)
        *   `tmax` (Maximum Temperature)
        *   `tmin` (Minimum Temperature)
        *   `streamflow` (Target variable, e.g., in CFS or mm/day)
3.  **Action:** Create a script to process your raw data into a basin-lumped NetCDF file (e.g., `target_basin.nc`).

### Step 2: Pre-train Base Model (CAMELS)
Train a single "Universal" LSTM on the 531 CAMELS basins to learn general physics.

1.  **Configuration (`camels_pretrain.yml`):**
    *   **Dataset:** `camels_us`
    *   **Forcing:** `daymet`
    *   **Inputs:** `prcp`, `tmax`, `tmin`
    *   **Target:** `qobs_mm_per_day`
    *   **Learning Rate:** Standard high rate (e.g., `1e-3`)
    *   **Epochs:** 30+
2.  **Execution:**
    ```bash
    nh-run train --config-file camels_pretrain.yml
    ```
3.  **Outcome:** A trained model directory in `runs/` (e.g., `runs/camels_pretrain_base_DATE`).

### Step 3: Fine-Tune on Target Basin
Adapt the universal model to your specific watershed and forcing data.

1.  **Configuration (`target_finetune.yml`):**
    *   **Experiment Name:** `target_finetune`
    *   **Base Run Directory:** Point to the folder created in Step 2.
        ```yaml
        base_run_dir: runs/camels_pretrain_base_DATE
        ```
    *   **Dataset:** `generic` (allows using your custom NetCDF)
    *   **Data Directory:** Path to your `target_basin.nc`.
    *   **Inputs:** Must match Base Model (`prcp`, `tmax`, `tmin`).
    *   **Learning Rate:** **Lower** than pre-training (e.g., `1e-4` or `5e-5`) to gently adapt weights without destroying learned physics.
    *   **Epochs:** Fewer required (e.g., 10-20).
2.  **Execution:**
    ```bash
    nh-run finetune --config-file target_finetune.yml
    ```

### Step 4: Validation & Inference
Evaluate the performance of the fine-tuned model.

1.  **Run Evaluation:**
    ```bash
    nh-run evaluate --run-dir runs/target_finetune_DATE
    ```
2.  **Metrics:** Compare NSE/KGE scores against baselines trained only on local data. Expect significantly higher performance (NSE > 0.90 is common) due to the transfer of hydrological knowledge.

---

## 4. Key Configuration Differences

| Feature | Base Model (Pre-training) | Target Model (Fine-tuning) |
| :--- | :--- | :--- |
| **Dataset Class** | `camels_us` | `generic` |
| **Forcing Source** | Daymet (Standard) | GridMET (Local Preference) |
| **Input Variables** | `prcp`, `tmax`, `tmin` | `prcp`, `tmax`, `tmin` |
| **Learning Rate** | `1e-3` (Fast learning) | `1e-4` (Gentle adaptation) |
| **Scope** | 531 diverse basins | 1 specific target basin |
| **Objective** | Learn General Physics | Learn Local Biases & Dynamics |
