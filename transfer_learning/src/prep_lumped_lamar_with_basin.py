import pandas as pd
import xarray as xr
import os
import numpy as np

def prep_lumped_data_with_basin():
    output_dir = 'transfer_learning/data/neuralhydrology/time_series'
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading data...")
    flow_path = 'transfer_learning/data/lamar_river_streamflow.csv'
    climate_path = 'transfer_learning/data/climate/lamar_gridmet_elevation_bands.csv'
    
    streamflow = pd.read_csv(flow_path, parse_dates=['date']).set_index('date')
    climate = pd.read_csv(climate_path, parse_dates=['date']).set_index('date')
    
    if streamflow.index.tz is not None:
        streamflow.index = streamflow.index.tz_localize(None)

    total_cells = 17 + 91 + 6
    weights = {
        'valley': 17 / total_cells,
        'mid': 91 / total_cells,
        'alpine': 6 / total_cells
    }
    
    df = pd.DataFrame(index=climate.index)
    df['prcp'] = (
        climate['precip_valley'] * weights['valley'] +
        climate['precip_mid'] * weights['mid'] +
        climate['precip_alpine'] * weights['alpine']
    )
    df['tmax'] = (
        climate['tmax_valley'] * weights['valley'] +
        climate['tmax_mid'] * weights['mid'] +
        climate['tmax_alpine'] * weights['alpine']
    )
    df['tmin'] = (
        climate['tmin_valley'] * weights['valley'] +
        climate['tmin_mid'] * weights['mid'] +
        climate['tmin_alpine'] * weights['alpine']
    )
    
    df = df.join(streamflow[['streamflow_cfs']], how='inner')
    
    # Convert to xarray with basin dimension
    ds = xr.Dataset.from_dataframe(df)
    
    # Add basin dimension
    basin_id = 'lamar_lumped'
    ds = ds.expand_dims(basin=[basin_id])
    
    # Note: NeuralHydrology basedataset.py usually calls to_dataframe()
    # If there's a 'basin' dimension, to_dataframe() creates a MultiIndex (basin, date).
    
    output_path = os.path.join(output_dir, 'lamar_lumped.nc')
    ds.to_netcdf(output_path)
    
    print(f"Saved Lumped Data to {output_path} with basin dimension.")
    print(f"Dimensions: {list(ds.dims)}")

if __name__ == "__main__":
    prep_lumped_data_with_basin()
