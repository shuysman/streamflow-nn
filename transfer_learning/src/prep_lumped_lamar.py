import pandas as pd
import os

def prep_lumped_data():
    output_dir = 'transfer_learning/data/neuralhydrology/time_series'
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading data...")
    flow_path = 'transfer_learning/data/lamar_river_streamflow.csv'
    climate_path = 'transfer_learning/data/climate/lamar_gridmet_elevation_bands.csv'
    # We can keep SNOTEL as extra features if we want, but CAMELS doesn't have SNOTEL usually.
    # To match CAMELS architecture perfectly, we should usually stick to Meteo forcing.
    # But we can add static attributes.
    
    # Load
    streamflow = pd.read_csv(flow_path, parse_dates=['date']).set_index('date')
    climate = pd.read_csv(climate_path, parse_dates=['date']).set_index('date')
    
    if streamflow.index.tz is not None:
        streamflow.index = streamflow.index.tz_localize(None)

    # Create Lumped Climate Features
    print("Aggregating elevation bands to lumped basin mean (Area Weighted)...")
    
    # Weights based on cell counts from download_gridmet_elevation_bands.py
    # Valley: 17 cells, Mid: 91 cells, Alpine: 6 cells
    total_cells = 17 + 91 + 6
    weights = {
        'valley': 17 / total_cells,
        'mid': 91 / total_cells,
        'alpine': 6 / total_cells
    }
    
    print(f"  Weights: Valley={weights['valley']:.2f}, Mid={weights['mid']:.2f}, Alpine={weights['alpine']:.2f}")

    df = pd.DataFrame(index=climate.index)
    
    # Calculate weighted averages
    # Precip
    df['prcp'] = (
        climate['precip_valley'] * weights['valley'] +
        climate['precip_mid'] * weights['mid'] +
        climate['precip_alpine'] * weights['alpine']
    )
    
    # Max Temp
    df['tmax'] = (
        climate['tmax_valley'] * weights['valley'] +
        climate['tmax_mid'] * weights['mid'] +
        climate['tmax_alpine'] * weights['alpine']
    )
    
    # Min Temp
    df['tmin'] = (
        climate['tmin_valley'] * weights['valley'] +
        climate['tmin_mid'] * weights['mid'] +
        climate['tmin_alpine'] * weights['alpine']
    )
    
    # Join Streamflow
    df = df.join(streamflow[['streamflow_cfs']], how='inner')
    
    # Save as NetCDF for NeuralHydrology
    # We need to reset index to get 'date' column for xarray conversion
    df_reset = df.reset_index()
    
    import xarray as xr
    ds = xr.Dataset.from_dataframe(df_reset.set_index('date'))
    
    output_path = os.path.join(output_dir, 'lamar_lumped.nc')
    ds.to_netcdf(output_path)
    
    print(f"Saved Lumped Data to {output_path}")
    print("Features: prcp, tmax, tmin, streamflow_cfs")

if __name__ == "__main__":
    prep_lumped_data()
