#!/usr/bin/env python3
"""
Download and process gridMET climate data for Hoh River watershed using elevation bands.

Adapts the Lamar approach for Hoh River:
- Downloads gridMET precipitation, temperature, solar radiation, and humidity data
- Converts specific humidity to vapor pressure using elevation-based pressure
- Classifies grid cells into 3 elevation bands (Valley, Mid, Alpine)
- Computes zonal statistics for each band
- Outputs daily time series suitable for LSTM input

Key difference from Lamar: Hoh is rain-dominated with lower elevation range.

Updated: 2025-12-23 - Added srad and vp for energy balance modeling
"""

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Elevation band thresholds for Hoh River (meters)
# Hoh is a rain-dominated coastal system with lower elevations than Lamar
BAND_THRESHOLDS = {
    'valley': (0, 800),        # < 2,625 ft: River valley, rain-dominated
    'mid': (800, 1500),        # 2,625-4,921 ft: Mid-elevation rain/snow transition
    'alpine': (1500, 2500)     # > 4,921 ft: Olympic peaks, snow accumulation
}


def calculate_surface_pressure(elevation_m):
    """
    Calculate surface pressure from elevation using barometric formula.
    P_surf = 101325 * (1 - 2.25577e-5 * Z)^5.25588
    """
    return 101325.0 * (1.0 - 2.25577e-5 * elevation_m) ** 5.25588


def specific_humidity_to_vapor_pressure(sph, elevation_m):
    """
    Convert specific humidity to vapor pressure.
    VP = (SPH * P_surf) / (0.622 + 0.378 * SPH)
    """
    p_surf = calculate_surface_pressure(elevation_m)
    sph = np.maximum(sph, 1e-10)
    vp = (sph * p_surf) / (0.622 + 0.378 * sph)
    return vp


def load_watershed():
    """Load the Hoh River watershed polygon."""
    watershed_path = Path('../data/watershed/hoh_watershed.geojson')
    if not watershed_path.exists():
        print(f"Error: Watershed file not found: {watershed_path}")
        print("Run fetch_hoh_watershed.py first!")
        return None

    gdf = gpd.read_file(watershed_path)
    print(f"✓ Loaded Hoh River watershed polygon")
    print(f"  Bounds: lon [{gdf.total_bounds[0]:.4f}, {gdf.total_bounds[2]:.4f}], "
          f"lat [{gdf.total_bounds[1]:.4f}, {gdf.total_bounds[3]:.4f}]")
    return gdf

def download_gridmet_opendap(start_date='1990-01-01', end_date='2025-01-01'):
    """
    Download gridMET data via OPeNDAP for Hoh River watershed.
    """
    print("\n" + "="*80)
    print("DOWNLOADING GRIDMET DATA FOR HOH RIVER")
    print("="*80)
    print(f"Date range: {start_date} to {end_date}")

    watershed = load_watershed()
    if watershed is None:
        return None

    bounds = watershed.total_bounds  # [minx, miny, maxx, maxy]
    print(f"\nQuerying gridMET for watershed region...")

    # gridMET OPeNDAP URLs
    base_url = "http://thredds.northwestknowledge.net:8080/thredds/dodsC/agg_met_"

    # Variables to download (using actual file names and variable names)
    variables = {
        'pr': ('pr_1979_CurrentYear_CONUS.nc', 'precipitation_amount'),
        'tmmn': ('tmmn_1979_CurrentYear_CONUS.nc', 'daily_minimum_temperature'),
        'tmmx': ('tmmx_1979_CurrentYear_CONUS.nc', 'daily_maximum_temperature'),
        'srad': ('srad_1979_CurrentYear_CONUS.nc', 'daily_mean_shortwave_radiation_at_surface'),
        'sph': ('sph_1979_CurrentYear_CONUS.nc', 'daily_mean_specific_humidity'),
    }

    datasets = {}

    for var_key, (filename, varname) in variables.items():
        url = base_url + filename
        print(f"\n  Accessing {var_key} ({varname})...")

        try:
            # Open dataset
            ds = xr.open_dataset(url, engine='netcdf4')

            # Subset by time first
            ds_time = ds.sel(day=slice(start_date, end_date))

            # Find grid cells within bounds
            lat_mask = (ds.lat >= bounds[1]) & (ds.lat <= bounds[3])
            lon_mask = (ds.lon >= bounds[0]) & (ds.lon <= bounds[2])

            lat_indices = np.where(lat_mask)[0]
            lon_indices = np.where(lon_mask)[0]

            print(f"    Found {len(lat_indices)} lat × {len(lon_indices)} lon grid cells")

            if len(lat_indices) == 0 or len(lon_indices) == 0:
                print(f"    WARNING: No grid cells found!")
                print(f"    Dataset bounds: lat [{float(ds.lat.min()):.2f}, {float(ds.lat.max()):.2f}], "
                      f"lon [{float(ds.lon.min()):.2f}, {float(ds.lon.max()):.2f}]")
                return None

            # Select subset
            ds_subset = ds_time.isel(lat=lat_indices, lon=lon_indices)

            # Load into memory
            print(f"    Loading data...")
            ds_loaded = ds_subset.load()

            # Store with standardized name
            datasets[var_key] = ds_loaded

            print(f"    ✓ {var_key}: shape {ds_loaded[varname].shape}")

        except Exception as e:
            print(f"    ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            return None

    # Merge datasets
    print(f"\nMerging datasets...")
    ds_merged = xr.Dataset({
        'precip': datasets['pr']['precipitation_amount'],
        'tmin': datasets['tmmn']['daily_minimum_temperature'],
        'tmax': datasets['tmmx']['daily_maximum_temperature'],
        'srad': datasets['srad']['daily_mean_shortwave_radiation_at_surface'],
        'sph': datasets['sph']['daily_mean_specific_humidity'],
    })

    print(f"✓ Merged dataset: {ds_merged.dims}")

    return ds_merged

def get_elevation_data(ds):
    """
    Get elevation data for each grid cell.
    Note: gridMET doesn't include elevation, so we'll use an external DEM.
    For now, we'll use a simplified approach based on latitude.
    """
    print("\n" + "="*80)
    print("ESTIMATING ELEVATION")
    print("="*80)

    # For Hoh River: elevation increases roughly with distance from coast
    # and with latitude (north = Olympic peaks)
    # This is a rough approximation - ideally use actual DEM

    # Coast is at ~-124.3°W, peaks are inland at ~-123.6°W
    # Elevation ranges from 0m (coast) to ~2400m (Mt. Olympus)

    lons = ds.lon.values
    lats = ds.lat.values

    # Create 2D mesh
    lon_mesh, lat_mesh = np.meshgrid(lons, lats)

    # Simple elevation model: increases with distance from coast and with latitude
    # Coast longitude: ~-124.3
    coast_lon = -124.3
    distance_from_coast = (lon_mesh - coast_lon) * 111  # degrees to km (approx)

    # Elevation increases ~100m per km inland (very rough!)
    elevation = np.clip(distance_from_coast * 100, 0, 2400)

    # Add elevation DataArray to dataset
    ds['elevation'] = xr.DataArray(
        elevation,
        dims=['lat', 'lon'],
        coords={'lat': lats, 'lon': lons},
        attrs={'units': 'meters', 'description': 'Estimated elevation (simplified model)'}
    )

    print(f"✓ Estimated elevation range: {elevation.min():.0f} - {elevation.max():.0f} m")
    print(f"  Note: This is a rough approximation. For production, use actual DEM data.")

    return ds

def classify_elevation_bands(ds):
    """
    Classify grid cells into elevation bands.
    """
    print("\n" + "="*80)
    print("CLASSIFYING ELEVATION BANDS")
    print("="*80)

    elevation = ds['elevation'].values

    # Create mask for each band
    bands = {}
    for band_name, (lower, upper) in BAND_THRESHOLDS.items():
        mask = (elevation >= lower) & (elevation < upper)
        n_cells = mask.sum()
        pct = (n_cells / mask.size) * 100

        bands[band_name] = mask

        print(f"  {band_name.capitalize():8s}: {lower:4.0f}-{upper:4.0f}m | "
              f"{n_cells:3d} cells ({pct:5.1f}%)")

    return bands

def compute_band_statistics(ds, bands):
    """
    Compute spatial average for each elevation band.
    """
    print("\n" + "="*80)
    print("COMPUTING ZONAL STATISTICS")
    print("="*80)

    results = {}

    for band_name, mask in bands.items():
        if mask.sum() == 0:
            print(f"  Warning: No cells in {band_name} band, skipping...")
            continue

        print(f"\n  Processing {band_name} band...")

        # Mask the data
        precip_masked = ds['precip'].where(mask)
        tmin_masked = ds['tmin'].where(mask)
        tmax_masked = ds['tmax'].where(mask)
        srad_masked = ds['srad'].where(mask)
        sph_masked = ds['sph'].where(mask)

        # Get mean elevation for this band for VP calculation
        elev_masked = ds['elevation'].where(mask)
        mean_elev = float(elev_masked.mean())

        # Compute spatial mean (over lat/lon)
        precip_mean = precip_masked.mean(dim=['lat', 'lon'])
        tmin_mean = tmin_masked.mean(dim=['lat', 'lon'])
        tmax_mean = tmax_masked.mean(dim=['lat', 'lon'])
        srad_mean = srad_masked.mean(dim=['lat', 'lon'])
        sph_mean = sph_masked.mean(dim=['lat', 'lon'])

        # Convert specific humidity to vapor pressure
        vp_mean = specific_humidity_to_vapor_pressure(sph_mean.values, mean_elev)

        # Convert to dataframe
        df = pd.DataFrame({
            f'precip_{band_name}': precip_mean.values,
            f'tmin_{band_name}': tmin_mean.values - 273.15,  # K to C
            f'tmax_{band_name}': tmax_mean.values - 273.15,
            f'srad_{band_name}': srad_mean.values,  # W/m2
            f'vp_{band_name}': vp_mean,  # Pa
        }, index=pd.to_datetime(ds['day'].values))

        # Derived variables
        df[f'temp_{band_name}'] = (df[f'tmin_{band_name}'] + df[f'tmax_{band_name}']) / 2

        results[band_name] = df

        print(f"    ✓ {len(df)} days")
        print(f"    Precip range: {df[f'precip_{band_name}'].min():.1f} - "
              f"{df[f'precip_{band_name}'].max():.1f} mm/day")
        print(f"    Temp range:   {df[f'temp_{band_name}'].min():.1f} - "
              f"{df[f'temp_{band_name}'].max():.1f} °C")

    # Combine all bands
    df_combined = pd.concat([results[band] for band in results.keys()], axis=1)
    df_combined.index.name = 'date'

    return df_combined

def main():
    """Main execution."""
    print("\n" + "="*80)
    print("HOH RIVER GRIDMET DOWNLOADER WITH ELEVATION BANDS")
    print("="*80)

    # Download gridMET data (full historical record)
    ds = download_gridmet_opendap(start_date='1990-01-01', end_date='2025-01-01')

    if ds is None:
        print("\n✗ Failed to download gridMET data")
        return

    # Get elevation data
    ds = get_elevation_data(ds)

    # Classify elevation bands
    bands = classify_elevation_bands(ds)

    # Compute statistics
    df = compute_band_statistics(ds, bands)

    # Save results
    output_file = '../data/climate/hoh_gridmet_elevation_bands.csv'
    df.to_csv(output_file)

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nRecords: {len(df)} days ({len(df)/365.25:.1f} years)")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Variables: {len(df.columns)} ({list(df.columns)})")
    print(f"Missing values: {df.isnull().sum().sum()} ({df.isnull().sum().sum() / df.size * 100:.2f}%)")

    print(f"\n✓ Data saved to: {output_file}")

    # Save metadata
    import json
    metadata = {
        'watershed': 'Hoh River near Forks, WA',
        'usgs_site': '12041200',
        'data_source': 'gridMET (4km resolution)',
        'elevation_bands': BAND_THRESHOLDS,
        'date_range': [str(df.index.min()), str(df.index.max())],
        'variables': list(df.columns),
        'records': len(df),
        'download_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    metadata_file = '../data/climate/hoh_gridmet_elevation_bands_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Metadata saved to: {metadata_file}")

    print("\n" + "="*80)
    print("✅ SUCCESS! Climate data ready for Stage 6")
    print("="*80)
    print("\nNext steps:")
    print("  1. Download SNOTEL data")
    print("  2. Update Stage 6 notebook to include climate features")

if __name__ == '__main__':
    main()
