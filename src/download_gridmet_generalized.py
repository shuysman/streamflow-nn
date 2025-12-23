#!/usr/bin/env python3
"""
Generalized GridMET Climate Data Downloader with Elevation Bands.

This script replaces basin-specific downloaders with a unified tool that:
1. Downloads gridMET data (precip, temp, srad, sph) via OPeNDAP.
2. Uses real Elevation data (metdata_elevationdata.nc) instead of proxies.
3. Computes both Zonal Statistics (Elevation Bands) AND Basin-Wide Means.
4. Outputs consistent CSV formats and Metadata JSONs.

Usage:
    python download_gridmet_generalized.py --watershed lamar
    python download_gridmet_generalized.py --watershed hoh
"""

import argparse
import json
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# Configuration Registry
# -----------------------------------------------------------------------------
WATERSHED_CONFIG = {
    'lamar': {
        'geojson': 'data/watershed/lamar_watershed.geojson',
        'output_csv': 'data/climate/lamar_gridmet_elevation_bands.csv',
        'output_json': 'data/climate/lamar_gridmet_elevation_bands_metadata.json',
        'usgs_id': '06188000',
        'bands': {
            'valley': (0, 2286),      # < 7,500 ft
            'mid': (2286, 2804),      # 7,500-9,200 ft
            'alpine': (2804, 10000)   # > 9,200 ft
        },
        'band_mean_elevations': {     # For Pressure calc fallback
            'valley': 2000,
            'mid': 2545,
            'alpine': 3100
        }
    },
    'hoh': {
        'geojson': 'data/watershed/hoh_watershed.geojson',
        'output_csv': 'data/climate/hoh_gridmet_elevation_bands.csv',
        'output_json': 'data/climate/hoh_gridmet_elevation_bands_metadata.json',
        'usgs_id': '12041200',
        'bands': {
            'valley': (0, 800),       # < 2,625 ft
            'mid': (800, 1500),       # 2,625-4,921 ft
            'alpine': (1500, 2500)    # > 4,921 ft
        },
        'band_mean_elevations': {
            'valley': 400,
            'mid': 1150,
            'alpine': 2000
        }
    }
}

# -----------------------------------------------------------------------------
# Physics Helper Functions
# -----------------------------------------------------------------------------
def calculate_surface_pressure(elevation_m):
    """P_surf = 101325 * (1 - 2.25577e-5 * Z)^5.25588"""
    return 101325.0 * (1.0 - 2.25577e-5 * elevation_m) ** 5.25588

def specific_humidity_to_vapor_pressure(sph, elevation_m):
    """VP = (SPH * P_surf) / (0.622 + 0.378 * SPH)"""
    p_surf = calculate_surface_pressure(elevation_m)
    sph = np.maximum(sph, 1e-10)
    vp = (sph * p_surf) / (0.622 + 0.378 * sph)
    return vp

# -----------------------------------------------------------------------------
# Core Processing Functions
# -----------------------------------------------------------------------------
def load_watershed(config):
    path = Path(config['geojson'])
    if not path.exists():
        raise FileNotFoundError(f"Watershed file not found: {path}")
    gdf = gpd.read_file(path)
    print(f"✓ Loaded watershed: {path.name}")
    return gdf

def download_gridmet_opendap(watershed, start_date, end_date):
    """Downloads gridMET data for the bounding box of the watershed."""
    print(f"\nDownloading GridMET ({start_date} to {end_date})...")
    bounds = watershed.total_bounds
    
    base_url = "http://thredds.northwestknowledge.net:8080/thredds/dodsC/agg_met_"
    variables = {
        'pr': ('pr_1979_CurrentYear_CONUS.nc', 'precipitation_amount'),
        'tmmn': ('tmmn_1979_CurrentYear_CONUS.nc', 'daily_minimum_temperature'),
        'tmmx': ('tmmx_1979_CurrentYear_CONUS.nc', 'daily_maximum_temperature'),
        'srad': ('srad_1979_CurrentYear_CONUS.nc', 'daily_mean_shortwave_radiation_at_surface'),
        'sph': ('sph_1979_CurrentYear_CONUS.nc', 'daily_mean_specific_humidity'),
    }

    datasets = []
    for output_name, (file_suffix, internal_var) in variables.items():
        try:
            ds = xr.open_dataset(base_url + file_suffix, engine='netcdf4')
            ds_time = ds.sel(day=slice(start_date, end_date))
            
            # Spatial Subset
            lat_mask = (ds.lat >= bounds[1]) & (ds.lat <= bounds[3])
            lon_mask = (ds.lon >= bounds[0]) & (ds.lon <= bounds[2])
            
            if not lat_mask.any() or not lon_mask.any():
                raise ValueError(f"No grid cells found for {output_name}")

            ds_subset = ds_time.isel(lat=np.where(lat_mask)[0], lon=np.where(lon_mask)[0])
            ds_loaded = ds_subset.load().rename({internal_var: output_name})
            datasets.append(ds_loaded)
            print(f"  ✓ {output_name}")
            
        except Exception as e:
            print(f"  ✗ Failed {output_name}: {e}")
            return None

    # Merge
    ds_merged = xr.merge(datasets)
    if 'day' in ds_merged.dims:
        ds_merged = ds_merged.rename({'day': 'time'})
    return ds_merged

def get_elevation_data(watershed, ref_ds):
    """Downloads real elevation data, aligned to the climate grid."""
    print("\nGetting Elevation Data...")
    try:
        url = "http://thredds.northwestknowledge.net:8080/thredds/dodsC/MET/elev/metdata_elevationdata.nc"
        ds_elev = xr.open_dataset(url, engine='netcdf4')
        
        # Subset to match reference climate data bounds
        ds_elev = ds_elev.sel(
            lat=slice(ref_ds.lat.max(), ref_ds.lat.min()), # GridMET lat is often inverted
            lon=slice(ref_ds.lon.min(), ref_ds.lon.max())
        )
        
        # Reindex to exact climate grid to ensure 1-to-1 matching
        ds_elev = ds_elev.reindex(lat=ref_ds.lat, lon=ref_ds.lon, method='nearest')
        return ds_elev['elevation'].load()
    except Exception as e:
        print(f"  ⚠ Elevation download failed ({e}). Using approximation.")
        # Fallback: simple gradient
        return None

def compute_statistics(ds, elevation, watershed, config):
    print("\nComputing Statistics...")
    
    # 1. Create Watershed Mask
    from shapely.geometry import Point
    geom = watershed.geometry.iloc[0]
    lons, lats = np.meshgrid(ds.lon, ds.lat)
    points = [Point(x, y) for x, y in zip(lons.ravel(), lats.ravel())]
    mask_flat = np.array([geom.contains(p) for p in points])
    watershed_mask = mask_flat.reshape(lons.shape)
    
    # 2. Setup Masks (Watershed + Elevation Bands)
    masks = {'basin': watershed_mask} # "basin" = global average
    
    if elevation is not None:
        elev_arr = elevation.values.squeeze() # Ensure 2D
        print(f"  Elevation Shape: {elev_arr.shape}")
        print(f"  Mask Shape:      {watershed_mask.shape}")
    else:
        # Fallback elevation
        elev_arr = np.full_like(watershed_mask, 1000.0, dtype=float) 

    for name, (low, high) in config['bands'].items():
        band_mask = (elev_arr >= low) & (elev_arr < high) & watershed_mask
        masks[name] = band_mask
        print(f"  Band {name}: {band_mask.sum()} cells")

    # 3. Iterate and Compute
    results = []
    times = ds.time.values
    total_steps = len(times)
    
    # Pre-fetch arrays to speed up loop
    pr = ds.pr.values
    tmmn = ds.tmmn.values
    tmmx = ds.tmmx.values
    srad = ds.srad.values
    sph = ds.sph.values
    
    print(f"  Processing {total_steps} time steps...")
    
    for i, t in enumerate(times):
        row = {'date': pd.to_datetime(t)}
        
        for name, mask in masks.items():
            if not mask.any():
                continue
                
            suffix = f"_{name}" if name != 'basin' else ""
            
            # Simple Means
            row[f'precip{suffix}'] = np.nanmean(pr[i][mask])
            row[f'tmin{suffix}'] = np.nanmean(tmmn[i][mask]) - 273.15
            row[f'tmax{suffix}'] = np.nanmean(tmmx[i][mask]) - 273.15
            row[f'srad{suffix}'] = np.nanmean(srad[i][mask])
            
            # Derived VP
            sph_mean = np.nanmean(sph[i][mask])
            # Use real mean elevation of the masked area for pressure calc
            if elevation is not None:
                mean_elev = np.nanmean(elev_arr[mask])
            else:
                mean_elev = config['band_mean_elevations'].get(name, 1000)
                
            row[f'vp{suffix}'] = specific_humidity_to_vapor_pressure(sph_mean, mean_elev)
            
        results.append(row)
        
        if i % 1000 == 0:
            print(f"    Step {i}/{total_steps}")

    return pd.DataFrame(results)

# -----------------------------------------------------------------------------
# Main Interface
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generalized GridMET Downloader")
    parser.add_argument("--watershed", choices=['lamar', 'hoh'], required=True, help="Target watershed")
    parser.add_argument("--start", default="1990-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2025-01-01", help="End date (YYYY-MM-DD)")
    args = parser.parse_args()

    config = WATERSHED_CONFIG[args.watershed]
    
    # 1. Load Geometry
    watershed = load_watershed(config)
    
    # 2. Download Climate
    ds = download_gridmet_opendap(watershed, args.start, args.end)
    if ds is None: return
    
    # 3. Get Elevation
    elev = get_elevation_data(watershed, ds)
    
    # 4. Compute Stats
    df = compute_statistics(ds, elev, watershed, config)
    
    # 5. Save
    print(f"\nSaving to {config['output_csv']}...")
    df.to_csv(config['output_csv'], index=False)
    
    # 6. Save Metadata
    meta = {
        'watershed': args.watershed,
        'config': config,
        'generated_at': str(datetime.now()),
        'records': len(df)
    }
    with open(config['output_json'], 'w') as f:
        json.dump(meta, f, indent=2)
        
    print("✓ Done.")

if __name__ == "__main__":
    main()
