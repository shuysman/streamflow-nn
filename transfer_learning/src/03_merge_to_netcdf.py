#!/usr/bin/env python3
"""
Stage 1.3: Merge to NetCDF Format

Combine climate + streamflow into NeuralHydrology-compatible NetCDF.

Creates xarray Dataset with:
- Variables: prcp_mm_day, tmax_C, tmin_C, QObs_mm_d
- Dimension: date (datetime64)
- No NaN values

Usage:
    python src/03_merge_to_netcdf.py --watershed lamar
    python src/03_merge_to_netcdf.py --watershed hoh
"""

import argparse
import pandas as pd
import xarray as xr
from pathlib import Path


def merge_to_netcdf(climate_file, streamflow_file, output_file):
    """
    Merge climate and streamflow data into NetCDF format.

    Args:
        climate_file: Path to lumped climate CSV
        streamflow_file: Path to streamflow mm/day CSV
        output_file: Path to output NetCDF file
    """
    print(f"\nReading climate data: {climate_file}")
    climate = pd.read_csv(climate_file, parse_dates=['date'])
    # Remove timezone if present
    if climate['date'].dt.tz is not None:
        climate['date'] = climate['date'].dt.tz_localize(None)
    print(f"  Climate records: {len(climate)}")
    print(f"  Date range: {climate['date'].min()} to {climate['date'].max()}")

    print(f"\nReading streamflow data: {streamflow_file}")
    streamflow = pd.read_csv(streamflow_file, parse_dates=['date'])
    # Remove timezone if present (to match climate data)
    if streamflow['date'].dt.tz is not None:
        streamflow['date'] = streamflow['date'].dt.tz_localize(None)
    print(f"  Streamflow records: {len(streamflow)}")
    print(f"  Date range: {streamflow['date'].min()} to {streamflow['date'].max()}")

    # Left join on climate (complete daily series) — streamflow will be NaN where missing
    print(f"\n=== Merging on Date ===")
    merged = climate.merge(streamflow, on='date', how='left')
    print(f"  Merged records: {len(merged)}")
    print(f"  Date range: {merged['date'].min()} to {merged['date'].max()}")

    # Check for missing values
    print(f"\n=== Pre-merge Validation ===")
    missing = merged.isnull().sum()
    print(f"Missing values:")
    print(missing)

    if missing['QObs_mm_d'] > 0:
        print(f"\n  Note: {int(missing['QObs_mm_d'])} days with missing streamflow (kept as NaN)")
        print(f"  NeuralHydrology will skip NaN targets in loss calculation")

    # Climate forcings should be complete
    climate_missing = missing.drop('QObs_mm_d').sum()
    assert climate_missing == 0, f"Missing climate forcing values: {missing}"

    assert len(merged) > 0, "No overlapping data between climate and streamflow!"

    # Create xarray Dataset
    print(f"\n=== Creating xarray Dataset ===")

    # Sort by date to ensure monotonic coordinate
    merged = merged.sort_values('date')

    ds = xr.Dataset(
        {
            'prcp_mm_day': (['date'], merged['prcp_mm_day'].values),
            'tmax_C': (['date'], merged['tmax_C'].values),
            'tmin_C': (['date'], merged['tmin_C'].values),
            'QObs_mm_d': (['date'], merged['QObs_mm_d'].values)
        },
        coords={'date': merged['date'].values}
    )

    # Add attributes for metadata
    ds.attrs['description'] = 'Basin-mean forcing and streamflow for NeuralHydrology'
    ds.attrs['created'] = pd.Timestamp.now().isoformat()

    ds['prcp_mm_day'].attrs['long_name'] = 'Precipitation'
    ds['prcp_mm_day'].attrs['units'] = 'mm/day'

    ds['tmax_C'].attrs['long_name'] = 'Maximum air temperature'
    ds['tmax_C'].attrs['units'] = 'degrees Celsius'

    ds['tmin_C'].attrs['long_name'] = 'Minimum air temperature'
    ds['tmin_C'].attrs['units'] = 'degrees Celsius'

    ds['QObs_mm_d'].attrs['long_name'] = 'Observed streamflow'
    ds['QObs_mm_d'].attrs['units'] = 'mm/day'

    # Validation
    print(f"\n=== Dataset Validation ===")
    print(f"Variables: {list(ds.data_vars)}")
    print(f"Coordinates: {list(ds.coords)}")
    print(f"Date dimension size: {len(ds.date)}")
    print(f"Date range: {ds.date.values[0]} to {ds.date.values[-1]}")

    # Check for NaN values
    nan_counts = {var: ds[var].isnull().sum().item() for var in ds.data_vars}
    print(f"\nNaN counts per variable:")
    for var, count in nan_counts.items():
        print(f"  {var}: {count}")

    # Climate forcings must be complete; streamflow NaNs are OK (skipped in loss)
    for var in ['prcp_mm_day', 'tmax_C', 'tmin_C']:
        assert nan_counts[var] == 0, f"NaN values in climate forcing {var}!"
    if nan_counts['QObs_mm_d'] > 0:
        print(f"  ({nan_counts['QObs_mm_d']} streamflow NaNs — OK, skipped in training loss)")
    else:
        print("✅ No NaN values")

    # Check data types
    print(f"\nData types:")
    for var in ds.data_vars:
        print(f"  {var}: {ds[var].dtype}")

    assert ds.date.dtype == 'datetime64[ns]', f"Date coord not datetime64! Got: {ds.date.dtype}"
    print("✅ Date coordinate is datetime64")

    # Save to NetCDF
    output_file.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(output_file)
    print(f"\n✅ Saved: {output_file}")

    # Verify file can be loaded
    print(f"\n=== Verification ===")
    ds_verify = xr.open_dataset(output_file)
    print(f"✅ File can be loaded")
    print(f"   Variables: {list(ds_verify.data_vars)}")
    print(f"   Shape: {ds_verify.dims}")
    ds_verify.close()

    print(f"\n✅ All checks passed")

    return ds


def main():
    parser = argparse.ArgumentParser(description='Merge climate and streamflow to NetCDF')
    parser.add_argument('--watershed', required=True, choices=['lamar', 'hoh'],
                        help='Watershed name')

    args = parser.parse_args()

    # Setup paths
    base_dir = Path(__file__).parent.parent
    climate_file = base_dir / f'data/processed/{args.watershed}_lumped.csv'
    streamflow_file = base_dir / f'data/processed/{args.watershed}_streamflow_mmday.csv'
    output_file = base_dir / f'data/neuralhydrology/time_series/{args.watershed}_lumped.nc'

    # Process
    merge_to_netcdf(climate_file, streamflow_file, output_file)


if __name__ == '__main__':
    main()
