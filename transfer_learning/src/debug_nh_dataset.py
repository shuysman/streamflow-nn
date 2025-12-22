from neuralhydrology.utils.config import Config
from neuralhydrology.datasetzoo.genericdataset import GenericDataset
from pathlib import Path

def debug_dataset():
    cfg = Config(Path('transfer_learning/config/lamar_finetune_config.yml'))
    # Mocking parts of the trainer to just get the dataset
    ds = GenericDataset(cfg=cfg, period="train", is_train=True)
    
    # Let's see what's in the xarray dataset loaded
    basin = ds.basins[0]
    xr_ds = ds._load_basin_data(basin)
    print("Xarray Dataset Variables:", list(xr_ds.data_vars))
    print("Xarray Dataset Coordinates:", list(xr_ds.coords))
    
    df = xr_ds.to_dataframe()
    print("DataFrame Columns:", df.columns.tolist())
    print("DataFrame Index:", df.index.names)

if __name__ == "__main__":
    try:
        debug_dataset()
    except Exception as e:
        print(f"Caught error: {e}")
        import traceback
        traceback.print_exc()
