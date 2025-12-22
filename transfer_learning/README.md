# Isolated Transfer Learning Implementation

This directory contains a clean, isolated implementation of the Transfer Learning workflow:
**CAMELS (General Hydrology)** $\to$ **Lamar River (Yellowstone Specific)**.

## Directory Structure
- `config/`: NeuralHydrology YAML configuration files.
- `src/`: Data preparation scripts modified for this isolated environment.
- `data/`: Local storage for CAMELS data and generated Lamar NetCDFs.

## Workflow Instructions

### Step 1: CAMELS Data (Prerequisite)
Due to licensing, you must manually place the CAMELS US dataset in:
`transfer_learning/data/CAMELS_US/`

Ensure the following folders exist within it:
- `basin_mean_forcing_daymet_v1.00/`
- `camels_attributes_v2.0/`
- `usgs_streamflow_v1.2/`

### Step 2: Prepare Lamar Target Data
Run the scripts in order to generate the lumped NetCDF file for Lamar:
```bash
python transfer_learning/src/download_gridmet_elevation_bands.py
python transfer_learning/src/prep_lumped_lamar.py
```
*Result:* `transfer_learning/data/neuralhydrology/time_series/lamar_lumped.nc`

### Step 3: Pre-train on CAMELS
This trains the "Universal" model on 531 basins.
```bash
nh-run train --config-file transfer_learning/config/camels_pretrain_config.yml
```
*Note:* This creates a folder in `runs/`. Note the name of this folder.

### Step 4: Fine-tune on Lamar
1. Open `transfer_learning/config/lamar_finetune_config.yml`.
2. Update `base_run_dir` to point to the run folder from Step 3.
3. Run the fine-tuning:
```bash
nh-run finetune --config-file transfer_learning/config/lamar_finetune_config.yml
```

## Benefits of this Approach
- **Climate Bias Handling:** Fine-tuning on GridMET after pre-training on Daymet automatically corrects for climate product biases.
- **Physical Realism:** Leverage the vast "knowledge" of the 531 CAMELS basins while specializing in the unique snow-melt dynamics of Lamar.

