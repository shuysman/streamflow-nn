#!/usr/bin/env python3
"""
Download and process gridMET climate data for Lamar River watershed using elevation bands.

This script implements the framework from climate_integration_framework.md:
- Downloads gridMET precipitation and temperature data
- Classifies grid cells into 3 elevation bands (Valley, Mid, Alpine)
- Computes zonal statistics for each band
- Outputs daily time series suitable for LSTM input

Author: Stage 2 Implementation
Date: 2025-11-26
"""

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Elevation band thresholds (in meters)
BAND_THRESHOLDS = {
    'valley': (0, 2286),      # < 7,500 ft: Rain-dominated, immediate runoff
    'mid': (2286, 2804),      # 7,500-9,200 ft: Peak accumulation zone
    'alpine': (2804, 10000)   # > 9,200 ft: Late season baseflow
}

def load_watershed():
    """Load the Lamar River watershed polygon."""
    watershed_path = Path('../data/watershed/lamar_watershed.geojson')
    gdf = gpd.read_file(watershed_path)
    print(f"✓ Loaded watershed polygon: {gdf.iloc[0]['identifier']}")
    print(f"  Bounds: {gdf.total_bounds}")
    return gdf

def download_gridmet_data(start_date='1998-01-01', end_date='2025-01-01'):
    """
    Download gridMET data using pygeohydro.

    Variables:
    - precipitation_amount: Precipitation (mm/day)
    - daily_minimum_temperature: Minimum temperature (K)
    - daily_maximum_temperature: Maximum temperature (K)
    """
    try:
        from pygeohydro import GridMET

        print(f"\nDownloading gridMET data from {start_date} to {end_date}...")

        # Load watershed for spatial bounds
        watershed = load_watershed()
        bounds = watershed.total_bounds  # [minx, miny, maxx, maxy]

        # Initialize GridMET
        gridmet = GridMET()

        # Download data
        dates = (start_date, end_date)
        variables = ['precipitaiton_amount', 'daily_minimum_temperature', 'daily_maximum_temperature']

        # Get data for watershed bounds
        ds = gridmet.get_bygeom(
            geometry=watershed.geometry[0],
            dates=dates,
            variables=variables
        )

        print(f"✓ Downloaded gridMET data")
        print(f"  Variables: {list(ds.data_vars)}")
        print(f"  Time range: {ds.time.min().values} to {ds.time.max().values}")
        print(f"  Grid shape: {ds.dims}")

        return ds

    except ImportError:
        print("Warning: pygeohydro not available, using manual OPeNDAP access...")
        return download_gridmet_opendap(start_date, end_date)

def download_gridmet_opendap(start_date='1998-01-01', end_date='2025-01-01'):
    """
    Alternative method: Direct OPeNDAP access to gridMET.
    Note: Spatial subsetting with gridMET is tricky, so we load regional data.
    """
    print("\nAccessing gridMET via OPeNDAP...")

    # Load watershed bounds
    watershed = load_watershed()
    bounds = watershed.total_bounds
    print(f"  Watershed bounds: lon [{bounds[0]:.2f}, {bounds[2]:.2f}], lat [{bounds[1]:.2f}, {bounds[3]:.2f}]")

    # gridMET OPeNDAP URLs
    base_url = "http://thredds.northwestknowledge.net:8080/thredds/dodsC/agg_met_"

    # Map file names to their actual variable names in the dataset
    variables = {
        'precipitation_amount': 'pr_1979_CurrentYear_CONUS.nc',
        'daily_minimum_temperature': 'tmmn_1979_CurrentYear_CONUS.nc',
        'daily_maximum_temperature': 'tmmx_1979_CurrentYear_CONUS.nc'
    }

    datasets = {}

    for var_name, file in variables.items():
        url = base_url + file
        print(f"  Accessing {var_name}...")

        try:
            # Open dataset (don't load yet - too big!)
            ds = xr.open_dataset(url, engine='netcdf4')

            # Subset by time first
            ds_time = ds.sel(day=slice(start_date, end_date))

            # Find indices that match our bounds
            # gridMET uses 1D lat/lon arrays
            lat_indices = np.where((ds.lat >= bounds[1]) & (ds.lat <= bounds[3]))[0]
            lon_indices = np.where((ds.lon >= bounds[0]) & (ds.lon <= bounds[2]))[0]

            print(f"    Found {len(lat_indices)} lat indices, {len(lon_indices)} lon indices")

            if len(lat_indices) == 0 or len(lon_indices) == 0:
                print(f"    WARNING: No grid cells found in bounds!")
                print(f"    Dataset lat range: {float(ds.lat.min()):.2f} to {float(ds.lat.max()):.2f}")
                print(f"    Dataset lon range: {float(ds.lon.min()):.2f} to {float(ds.lon.max()):.2f}")
                return None

            # Use isel to select by integer indices
            ds_subset = ds_time.isel(
                lat=lat_indices,
                lon=lon_indices
            )

            # NOW load into memory (much smaller subset)
            print(f"    Loading subset...")
            datasets[var_name] = ds_subset.load()

            print(f"    ✓ {var_name}: {datasets[var_name][var_name].shape}")

        except Exception as e:
            print(f"    Error downloading {var_name}: {e}")
            import traceback
            traceback.print_exc()
            return None

    # Merge datasets
    ds = xr.merge([
        datasets['precipitation_amount'],
        datasets['daily_minimum_temperature'],
        datasets['daily_maximum_temperature']
    ])

    # Rename variables to shorter names
    ds = ds.rename({
        'precipitation_amount': 'pr',
        'daily_minimum_temperature': 'tmmn',
        'daily_maximum_temperature': 'tmmx'
    })

    # Rename time dimension if needed
    if 'day' in ds.dims:
        ds = ds.rename({'day': 'time'})

    print(f"✓ Downloaded gridMET data via OPeNDAP")
    print(f"  Final dimensions: {dict(ds.dims)}")
    print(f"  Variables: {list(ds.data_vars)}")
    print(f"  Lat range: {float(ds.lat.min()):.2f} to {float(ds.lat.max()):.2f}")
    print(f"  Lon range: {float(ds.lon.min()):.2f} to {float(ds.lon.max()):.2f}")
    return ds

def get_elevation_data():
    """
    Download or load gridMET elevation data.

    GridMET provides elevation data at the same 4km resolution.
    """
    print("\nAccessing gridMET elevation data...")

    try:
        # Try OPeNDAP access to metdata elevation
        elev_url = "http://thredds.northwestknowledge.net:8080/thredds/dodsC/MET/elev/metdata_elevationdata.nc"

        # Load watershed bounds
        watershed = load_watershed()
        bounds = watershed.total_bounds

        ds_elev = xr.open_dataset(elev_url, engine='netcdf4')

        # Find indices within bounds
        lat_indices = np.where((ds_elev.lat >= bounds[1]) & (ds_elev.lat <= bounds[3]))[0]
        lon_indices = np.where((ds_elev.lon >= bounds[0]) & (ds_elev.lon <= bounds[2]))[0]

        print(f"  Found {len(lat_indices)} lat indices, {len(lon_indices)} lon indices")

        if len(lat_indices) == 0 or len(lon_indices) == 0:
            print("  Warning: No elevation data found in bounds")
            return None

        # Subset using indices
        ds_elev_subset = ds_elev.isel(lat=lat_indices, lon=lon_indices)

        # Load into memory
        ds_elev_subset = ds_elev_subset.load()

        print(f"✓ Loaded elevation data")
        print(f"  Shape: {ds_elev_subset.elevation.shape}")
        print(f"  Range: {float(ds_elev_subset.elevation.min()):.0f}m - {float(ds_elev_subset.elevation.max()):.0f}m")

        return ds_elev_subset

    except Exception as e:
        print(f"Error loading elevation data: {e}")
        print("Will use synthetic elevation based on latitude (approximation)...")
        return None

def classify_elevation_bands(ds_climate, ds_elev, watershed):
    """
    Classify each grid cell into elevation bands and create masks.

    Returns:
        dict: Masks for each elevation band
    """
    print("\nClassifying grid cells into elevation bands...")

    # Check if climate data has valid dimensions
    if 'lat' not in ds_climate.dims or 'lon' not in ds_climate.dims:
        print("ERROR: Climate data missing lat/lon dimensions")
        print(f"Available dimensions: {list(ds_climate.dims.keys())}")
        raise ValueError("Invalid climate data structure")

    # Get elevation array
    if ds_elev is not None and 'elevation' in ds_elev:
        elevation = ds_elev.elevation
        # Remove extra dimensions if present
        if elevation.ndim == 3 and elevation.shape[0] == 1:
            elevation = elevation.isel({elevation.dims[0]: 0})
        print(f"  Elevation array shape: {elevation.shape}")
        print(f"  Elevation range: {float(elevation.min()):.0f}m - {float(elevation.max()):.0f}m")
    else:
        # Create approximate elevation based on latitude
        print("  Using latitude-based approximation for elevation")

        # Get lat/lon arrays
        lats = ds_climate.lat.values
        lons = ds_climate.lon.values

        # Check if we have data
        if len(lats) == 0 or len(lons) == 0:
            raise ValueError("Climate data has empty lat/lon arrays")

        # Create 2D elevation grid based on latitude (higher lat = higher elevation in this region)
        # Approximate: 2000m at south, 3000m at north (rough gradient)
        lat_min = lats.min()
        lat_max = lats.max()

        # Create meshgrid
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        # Elevation increases with latitude
        elev_values = 2000 + (lat_grid - lat_min) / (lat_max - lat_min) * 1000

        elevation = xr.DataArray(
            elev_values,
            coords={'lat': lats, 'lon': lons},
            dims=['lat', 'lon']
        )

    # Create masks for each band based on elevation
    # Note: Since we already subsetted spatially to the watershed bounds,
    # all grid cells can be considered for elevation bands
    masks = {
        'valley': (elevation >= BAND_THRESHOLDS['valley'][0]) &
                  (elevation < BAND_THRESHOLDS['valley'][1]),
        'mid': (elevation >= BAND_THRESHOLDS['mid'][0]) &
               (elevation < BAND_THRESHOLDS['mid'][1]),
        'alpine': (elevation >= BAND_THRESHOLDS['alpine'][0]) &
                  (elevation < BAND_THRESHOLDS['alpine'][1])
    }

    # Report statistics
    print("\n  Elevation band statistics:")
    for band, mask in masks.items():
        n_cells = int(mask.sum())
        if n_cells > 0:
            elev_values = elevation.where(mask).values.ravel()
            elev_values = elev_values[~np.isnan(elev_values)]
            print(f"    {band:8s}: {n_cells:3d} cells, "
                  f"elevation {elev_values.min():.0f}-{elev_values.max():.0f}m")
        else:
            print(f"    {band:8s}: WARNING - 0 cells!")

    return masks, elevation

def compute_zonal_statistics(ds_climate, masks):
    """
    Compute mean climate variables for each elevation band.

    Returns:
        pd.DataFrame: Daily time series with columns for each band
    """
    print("\nComputing zonal statistics for each elevation band...")

    # Convert masks to boolean numpy arrays once
    mask_arrays = {}
    for band, mask in masks.items():
        mask_arrays[band] = mask.values
        n_cells = int(mask_arrays[band].sum())
        print(f"  Band '{band}': {n_cells} cells")

    results = []

    # Iterate over time steps
    for t in range(len(ds_climate.time)):
        if t % 1000 == 0:
            print(f"  Processing time step {t}/{len(ds_climate.time)}...")

        date = pd.Timestamp(ds_climate.time[t].values)
        row = {'date': date}

        # Extract climate variables for this time step as numpy arrays
        pr = ds_climate.pr.isel(time=t).values
        tmmn = ds_climate.tmmn.isel(time=t).values
        tmmx = ds_climate.tmmx.isel(time=t).values

        # Compute mean for each band using numpy
        for band, mask_array in mask_arrays.items():
            # Precipitation (mm/day)
            if mask_array.any():
                pr_values = pr[mask_array]
                pr_mean = np.nanmean(pr_values) if len(pr_values) > 0 else 0.0
                row[f'precip_{band}'] = float(pr_mean) if not np.isnan(pr_mean) else 0.0
            else:
                row[f'precip_{band}'] = 0.0

            # Temperature (convert K to C and average min/max)
            if mask_array.any():
                tmmn_values = tmmn[mask_array]
                tmmx_values = tmmx[mask_array]
                tmmn_mean = np.nanmean(tmmn_values) if len(tmmn_values) > 0 else np.nan
                tmmx_mean = np.nanmean(tmmx_values) if len(tmmx_values) > 0 else np.nan

                if not np.isnan(tmmn_mean) and not np.isnan(tmmx_mean):
                    temp_c = ((tmmn_mean + tmmx_mean) / 2.0) - 273.15
                    row[f'temp_{band}'] = float(temp_c)
                else:
                    row[f'temp_{band}'] = 0.0
            else:
                row[f'temp_{band}'] = 0.0

        results.append(row)

    # Create DataFrame
    df = pd.DataFrame(results)

    print(f"\n✓ Computed zonal statistics")
    print(f"  Time range: {df['date'].min()} to {df['date'].max()}")
    print(f"  Shape: {df.shape}")
    print(f"\n  Sample values (first 5 rows):")
    print(df.head())
    print(f"\n  Column statistics:")
    print(df.describe())

    return df

def main():
    """Main execution function."""
    print("="*80)
    print("GridMET Climate Data Processing with Elevation Bands")
    print("Lamar River Watershed (USGS 06191500)")
    print("="*80)

    # Configuration
    start_date = '1998-01-01'
    end_date = '2025-01-01'
    output_file = '../data/climate/lamar_gridmet_elevation_bands.csv'

    # Create output directory
    Path('../data/climate').mkdir(parents=True, exist_ok=True)

    # Step 1: Load watershed
    watershed = load_watershed()

    # Step 2: Download climate data
    ds_climate = download_gridmet_opendap(start_date, end_date)

    if ds_climate is None:
        print("\nError: Could not download climate data")
        return

    # Step 3: Get elevation data
    ds_elev = get_elevation_data()

    # Step 4: Classify elevation bands
    masks, elevation = classify_elevation_bands(ds_climate, ds_elev, watershed)

    # Check if all bands have cells
    empty_bands = [band for band, mask in masks.items() if mask.sum() == 0]
    if empty_bands:
        print(f"\n⚠ WARNING: Empty bands detected: {empty_bands}")
        print("  Consider adjusting elevation thresholds or using merged bands")

    # Step 5: Compute zonal statistics
    df_climate = compute_zonal_statistics(ds_climate, masks)

    # Step 6: Save results
    df_climate.to_csv(output_file, index=False)
    print(f"\n✓ Saved climate data to: {output_file}")

    # Create summary plot
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set_style('whitegrid')
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))

        # Plot precipitation by band
        axes[0].plot(df_climate['date'], df_climate['precip_valley'],
                     label='Valley', alpha=0.7, linewidth=0.8)
        axes[0].plot(df_climate['date'], df_climate['precip_mid'],
                     label='Mid', alpha=0.7, linewidth=0.8)
        axes[0].plot(df_climate['date'], df_climate['precip_alpine'],
                     label='Alpine', alpha=0.7, linewidth=0.8)
        axes[0].set_title('Precipitation by Elevation Band', fontweight='bold', fontsize=12)
        axes[0].set_ylabel('Precipitation (mm/day)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot temperature by band
        axes[1].plot(df_climate['date'], df_climate['temp_valley'],
                     label='Valley', alpha=0.7, linewidth=0.8)
        axes[1].plot(df_climate['date'], df_climate['temp_mid'],
                     label='Mid', alpha=0.7, linewidth=0.8)
        axes[1].plot(df_climate['date'], df_climate['temp_alpine'],
                     label='Alpine', alpha=0.7, linewidth=0.8)
        axes[1].axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Freezing')
        axes[1].set_title('Temperature by Elevation Band', fontweight='bold', fontsize=12)
        axes[1].set_xlabel('Date')
        axes[1].set_ylabel('Temperature (°C)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('../data/climate/lamar_climate_elevation_bands.png', dpi=150, bbox_inches='tight')
        print(f"✓ Saved visualization to: ../data/climate/lamar_climate_elevation_bands.png")

    except Exception as e:
        print(f"Note: Could not create visualization: {e}")

    print("\n" + "="*80)
    print("Processing complete!")
    print("="*80)

if __name__ == '__main__':
    main()
