#!/usr/bin/env python3
"""
Download point-scale GridMET climate data for SNOTEL stations.

This script fetches daily precipitation and temperature data from GridMET
specifically for the pixel containing each SNOTEL station. This enables
"Point-to-Point" calibration where we match the physics of the location.

Stations:
- Parker Peak (683): 44.73396 N, -109.91484 W
- Northeast Entrance (670): 45.00000 N, -110.00000 W

Author: Gemini Agent
Date: 2025-12-15
"""

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
import warnings
import time

warnings.filterwarnings('ignore')

# Station configuration
STATIONS = {
    'parker_peak': {
        'id': '683',
        'lat': 44.73396,
        'lon': -109.91484,
        'elevation_m': 2871
    },
    'northeast_entrance': {
        'id': '670',
        'lat': 45.00000,
        'lon': -110.0167,
        'elevation_m': 2262
    }
}

def download_gridmet_point(lat, lon, station_name, start_date='1998-01-01', end_date='2025-01-01'):
    """
    Download GridMET data for a specific point via OPeNDAP.
    """
    print(f"\nProcessing {station_name} ({lat}, {lon})...")

    # gridMET OPeNDAP URLs
    base_url = "http://thredds.northwestknowledge.net:8080/thredds/dodsC/agg_met_"
    
    variables = {
        'precipitation_amount': 'pr_1979_CurrentYear_CONUS.nc',
        'daily_minimum_temperature': 'tmmn_1979_CurrentYear_CONUS.nc',
        'daily_maximum_temperature': 'tmmx_1979_CurrentYear_CONUS.nc'
    }

    results = {}
    
    # GridMET uses 1D lat/lon arrays (usually)
    # We need to find the nearest index for our point
    
    for var_name, file in variables.items():
        url = base_url + file
        print(f"  Accessing {var_name}...")
        
        try:
            ds = xr.open_dataset(url, engine='netcdf4')
            
            # Find nearest lat/lon index
            # Note: GridMET coords are often named 'lat' and 'lon'
            lat_idx = abs(ds.lat - lat).argmin().values
            lon_idx = abs(ds.lon - lon).argmin().values
            
            # Verify distance (GridMET is ~4km resolution, so diff should be small)
            grid_lat = float(ds.lat[lat_idx].values)
            grid_lon = float(ds.lon[lon_idx].values)
            dist_sq = (grid_lat - lat)**2 + (grid_lon - lon)**2
            
            print(f"    Nearest grid point: {grid_lat:.4f}, {grid_lon:.4f} (diff: {np.sqrt(dist_sq):.4f})")
            
            # Select point and time range
            ds_point = ds.isel(lat=lat_idx, lon=lon_idx).sel(day=slice(start_date, end_date))
            
            # Load into memory
            results[var_name] = ds_point.load()
            
        except Exception as e:
            print(f"    Error downloading {var_name}: {e}")
            return None

    # Process into DataFrame
    print("  Processing time series...")
    
    # Extract time array from one variable
    times = results['precipitation_amount'].day.values
    dates = pd.to_datetime(times)
    
    df = pd.DataFrame({'date': dates})
    
    # Extract values
    # Precip is in mm
    df['precip'] = results['precipitation_amount']['precipitation_amount'].values
    
    # Temp is in Kelvin
    tmmn_k = results['daily_minimum_temperature']['daily_minimum_temperature'].values
    tmmx_k = results['daily_maximum_temperature']['daily_maximum_temperature'].values
    
    # Convert to C and calculate mean
    df['tmin'] = tmmn_k - 273.15
    df['tmax'] = tmmx_k - 273.15
    df['temp'] = (df['tmin'] + df['tmax']) / 2.0
    
    # Clean up (GridMET uses -9999 for missing?)
    # Usually xarray handles fill values, but just in case
    
    return df

def main():
    print("="*80)
    print("SNOTEL Point-Scale Climate Data Downloader")
    print("="*80)
    
    output_dir = Path('data/climate')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    start_date = '1998-01-01'
    end_date = '2025-01-01'
    
    for name, info in STATIONS.items():
        df = download_gridmet_point(
            info['lat'], 
            info['lon'], 
            name,
            start_date, 
            end_date
        )
        
        if df is not None:
            output_file = output_dir / f"snotel_gridmet_{name}.csv"
            df.to_csv(output_file, index=False)
            print(f"  ✓ Saved to {output_file}")
            print(f"  Rows: {len(df)}")
            print(f"  Precip Mean: {df['precip'].mean():.2f} mm")
            print(f"  Temp Mean: {df['temp'].mean():.2f} C")
        
        # Pause to be nice to server
        time.sleep(2)
        
    print("\n" + "="*80)
    print("Processing complete!")

if __name__ == '__main__':
    main()
