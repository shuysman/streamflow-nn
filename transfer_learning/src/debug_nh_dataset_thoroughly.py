from neuralhydrology.utils.config import Config
from neuralhydrology.datasetzoo.genericdataset import GenericDataset
from pathlib import Path
import pandas as pd
import xarray as xr

def debug_dataset_thoroughly():
    # Load config exactly as nh-run does (merging with base run)
    # We can't easily merge here, so we just use the fine-tune config
    # and add missing pieces.
    cfg_path = 'transfer_learning/config/lamar_finetune_config.yml'
    cfg = Config(Path(cfg_path))
    
    # GenericDataset load_basin_data
    from neuralhydrology.datasetzoo.genericdataset import load_basin_data
    xr_ds = load_basin_data(cfg.data_dir, 'lamar_lumped')
    
    print("Xarray Dataset:")
    print(xr_ds)
    
    df_native = xr_ds.to_dataframe()
    print("\nDataFrame Native Columns:")
    print(df_native.columns)
    print("\nDataFrame Native Index:")
    print(df_native.index)
    
    dynamic_cols = cfg.dynamic_inputs
    target_vars = cfg.target_variables
    
    print(f"\nDynamic Inputs from config: {dynamic_cols}")
    print(f"Target Variables from config: {target_vars}")
    
    # Try the problematic line
    cols_to_select = dynamic_cols + target_vars
    print(f"\nAttempting to select columns: {cols_to_select}")
    
    try:
        df_sub = df_native[cols_to_select]
        print("Selection successful!")
    except Exception as e:
        print(f"Selection failed with error: {e}")
        # Let's inspect types
        print(f"Type of cols_to_select: {type(cols_to_select)}")
        for i, col in enumerate(cols_to_select):
            print(f"  Col {i}: '{col}' (Type: {type(col)})")
        
        # Check if column names have hidden whitespace
        print("\nColumn names in DF (quoted):")
        for col in df_native.columns:
            print(f"  '{col}'")

if __name__ == "__main__":
    debug_dataset_thoroughly()
