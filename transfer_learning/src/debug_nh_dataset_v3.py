from neuralhydrology.utils.config import Config
from neuralhydrology.datasetzoo.genericdataset import GenericDataset
from pathlib import Path
import pandas as pd
import xarray as xr
import os

def debug_dataset_thoroughly():
    # Setup paths for manual load if NH load fails
    data_dir = Path('transfer_learning/data/neuralhydrology')
    basin = 'lamar_lumped'
    nc_path = data_dir / "time_series" / f"{basin}.nc"
    
    print(f"Opening {nc_path} manually...")
    xr_ds = xr.open_dataset(nc_path)
    
    print("Xarray Dataset Variables:", list(xr_ds.data_vars))
    print("Xarray Dataset Coordinates:", list(xr_ds.coords))
    
    df_native = xr_ds.to_dataframe()
    print("\nDataFrame Columns:", df_native.columns.tolist())
    
    # Try the NH config access
    cfg_path = 'transfer_learning/config/lamar_finetune_config.yml'
    cfg = Config(Path(cfg_path))
    
    dynamic_cols = cfg.dynamic_inputs
    target_vars = cfg.target_variables
    
    print(f"\nDynamic Inputs from config: {dynamic_cols}")
    print(f"Target Variables from config: {target_vars}")
    
    # Check if they are lists or something else
    print(f"Type of dynamic_cols: {type(dynamic_cols)}")
    
    # Simulate the addition
    all_cols = dynamic_cols + target_vars
    print(f"All cols sum: {all_cols}")
    print(f"Type of sum: {type(all_cols)}")
    
    try:
        df_sub = df_native[all_cols]
        print("Selection successful!")
    except Exception as e:
        print(f"Selection failed: {e}")

if __name__ == "__main__":
    debug_dataset_thoroughly()
