import pandas as pd
import xarray as xr
import os

def prep_lumped_data_camels_names():
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
    
    # Use CAMELS names
    df['prcp(mm/day)'] = (
        climate['precip_valley'] * weights['valley'] +
        climate['precip_mid'] * weights['mid'] +
        climate['precip_alpine'] * weights['alpine']
    )
    df['tmax(C)'] = (
        climate['tmax_valley'] * weights['valley'] +
        climate['tmax_mid'] * weights['mid'] +
        climate['tmax_alpine'] * weights['alpine']
    )
    df['tmin(C)'] = (
        climate['tmin_valley'] * weights['valley'] +
        climate['tmin_mid'] * weights['mid'] +
        climate['tmin_alpine'] * weights['alpine']
    )
    
    # Join Streamflow - keeping local name for now, but will map in config
    df = df.join(streamflow[['streamflow_cfs']], how='inner')
    
    df_reset = df.reset_index()
    ds = xr.Dataset.from_dataframe(df_reset.set_index('date'))
    
    output_path = os.path.join(output_dir, 'lamar_lumped.nc')
    ds.to_netcdf(output_path)
    
    print(f"Saved Lumped Data to {output_path}")
    print(f"Features: {list(ds.data_vars)}")

if __name__ == "__main__":
    prep_lumped_data_camels_names()
