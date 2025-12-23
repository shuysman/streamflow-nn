import pandas as pd
import xarray as xr
import numpy as np
import os

def prep_lumped_data_final():
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
    ).astype('float32')
    df['tmax'] = (
        climate['tmax_valley'] * weights['valley'] +
        climate['tmax_mid'] * weights['mid'] +
        climate['tmax_alpine'] * weights['alpine']
    ).astype('float32')
    df['tmin'] = (
        climate['tmin_valley'] * weights['valley'] +
        climate['tmin_mid'] * weights['mid'] +
        climate['tmin_alpine'] * weights['alpine']
    ).astype('float32')
    
    df = df.join(streamflow[['streamflow_cfs']], how='inner')
    df['streamflow_cfs'] = df['streamflow_cfs'].astype('float32')
    
    # Create xarray from scratch to avoid any pandas meta-data carryover
    ds = xr.Dataset(
        data_vars={
            'prcp': (['date'], df['prcp'].values),
            'tmax': (['date'], df['tmax'].values),
            'tmin': (['date'], df['tmin'].values),
            'streamflow_cfs': (['date'], df['streamflow_cfs'].values),
        },
        coords={
            'date': df.index.values
        }
    )
    
    output_path = os.path.join(output_dir, 'lamar_lumped.nc')
    # Use simple netcdf engine
    ds.to_netcdf(output_path)
    
    print(f"Saved Lumped Data to {output_path}")

if __name__ == "__main__":
    prep_lumped_data_final()
