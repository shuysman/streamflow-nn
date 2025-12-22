import xarray as xr
import numpy as np
import os

def inject_statics():
    nc_path = 'transfer_learning/data/neuralhydrology/time_series/lamar_lumped.nc'
    if not os.path.exists(nc_path):
        print(f"Error: {nc_path} not found.")
        return

    print(f"Opening {nc_path}...")
    ds = xr.open_dataset(nc_path)
    
    # Static attributes to match CAMELS pre-training
    # Approximate values for Lamar River (Yellowstone)
    statics = {
        'elev_mean': 2600.0,
        'slope_mean': 30.0,
        'area_gages2': 1730.0,
        'frac_forest': 0.6,
        'soil_depth_pelletier': 1.0,
        'sand_frac': 0.3,
        'clay_frac': 0.3
    }
    
    print("Injecting static attributes...")
    for key, val in statics.items():
        # NeuralHydrology expects statics to be variables in the NetCDF
        # We broadcast the scalar value across the 'date' dimension
        ds[key] = (('date'), np.full(len(ds.date), val))
        
    ds.to_netcdf(nc_path + '.tmp')
    ds.close()
    
    os.replace(nc_path + '.tmp', nc_path)
    print(f"Successfully updated {nc_path} with static attributes.")

if __name__ == "__main__":
    inject_statics()
