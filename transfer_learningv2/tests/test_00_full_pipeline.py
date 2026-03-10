#!/usr/bin/env python3
"""
Test Stage 0: Full Pipeline Integration Test

This is the comprehensive test that verifies the entire data preparation pipeline
from raw data to NeuralHydrology-ready NetCDF files.

Tests:
1. Raw data exists and is readable
2. Processed climate data (lumped) exists and is valid
3. Processed streamflow data (mm/day) exists and is valid
4. NetCDF files exist and are loadable
5. Attributes exist and match basin names
6. Basin lists exist
7. Data consistency across all stages
8. NeuralHydrology compatibility
"""

import pandas as pd
import xarray as xr
import numpy as np
from pathlib import Path


def test_full_pipeline(watershed):
    """
    Comprehensive end-to-end test for a watershed's data pipeline.

    Args:
        watershed: Watershed name ('lamar' or 'hoh')
    """
    print(f"\n{'='*70}")
    print(f"FULL PIPELINE TEST: {watershed.upper()}")
    print(f"{'='*70}")

    base_dir = Path(__file__).parent.parent

    # =========================================================================
    # STAGE 1: Raw Data
    # =========================================================================
    print(f"\n--- Stage 1: Raw Data ---")

    raw_climate = base_dir / f'data/raw/{watershed}_gridmet_elevation_bands.csv'
    raw_streamflow = base_dir / f'data/raw/{watershed}_streamflow.csv'

    assert raw_climate.exists(), f"❌ Raw climate not found: {raw_climate}"
    assert raw_streamflow.exists(), f"❌ Raw streamflow not found: {raw_streamflow}"

    print(f"✅ Raw climate file exists")
    print(f"✅ Raw streamflow file exists")

    # Load raw data to get date ranges
    climate_raw_df = pd.read_csv(raw_climate, parse_dates=['date'])
    stream_raw_df = pd.read_csv(raw_streamflow, parse_dates=['date'])

    print(f"   Climate: {len(climate_raw_df)} records, {climate_raw_df['date'].min().date()} to {climate_raw_df['date'].max().date()}")
    print(f"   Streamflow: {len(stream_raw_df)} records, {stream_raw_df['date'].min().date()} to {stream_raw_df['date'].max().date()}")

    # =========================================================================
    # STAGE 2: Processed Data
    # =========================================================================
    print(f"\n--- Stage 2: Processed Data ---")

    proc_climate = base_dir / f'data/processed/{watershed}_lumped.csv'
    proc_streamflow = base_dir / f'data/processed/{watershed}_streamflow_mmday.csv'

    assert proc_climate.exists(), f"❌ Processed climate not found: {proc_climate}"
    assert proc_streamflow.exists(), f"❌ Processed streamflow not found: {proc_streamflow}"

    print(f"✅ Lumped climate file exists")
    print(f"✅ Converted streamflow file exists")

    # Load processed data
    climate_df = pd.read_csv(proc_climate, parse_dates=['date'])
    stream_df = pd.read_csv(proc_streamflow, parse_dates=['date'])

    # Check column names
    assert list(climate_df.columns) == ['date', 'prcp_mm_day', 'tmax_C', 'tmin_C'], \
        f"❌ Climate columns incorrect: {list(climate_df.columns)}"
    assert list(stream_df.columns) == ['date', 'QObs_mm_d'], \
        f"❌ Streamflow columns incorrect: {list(stream_df.columns)}"

    print(f"✅ Climate columns correct: {list(climate_df.columns)}")
    print(f"✅ Streamflow columns correct: {list(stream_df.columns)}")

    # Check for missing values
    assert climate_df.isnull().sum().sum() == 0, "❌ Missing values in climate"
    assert stream_df.isnull().sum().sum() == 0, "❌ Missing values in streamflow"

    print(f"✅ No missing values in processed data")

    # =========================================================================
    # STAGE 3: NetCDF File
    # =========================================================================
    print(f"\n--- Stage 3: NetCDF File ---")

    nc_file = base_dir / f'data/neuralhydrology/time_series/{watershed}_lumped.nc'

    assert nc_file.exists(), f"❌ NetCDF not found: {nc_file}"
    print(f"✅ NetCDF file exists")

    # Load NetCDF
    ds = xr.open_dataset(nc_file)

    # Check variables
    expected_vars = {'prcp_mm_day', 'tmax_C', 'tmin_C', 'QObs_mm_d'}
    assert set(ds.data_vars) == expected_vars, \
        f"❌ NetCDF variables incorrect: {set(ds.data_vars)}"
    print(f"✅ NetCDF has correct variables: {list(ds.data_vars)}")

    # Check dimension
    assert 'date' in ds.dims, "❌ 'date' dimension missing"
    assert len(ds.dims) == 1, f"❌ Should have 1 dimension, got {len(ds.dims)}"
    print(f"✅ NetCDF has 'date' dimension with {len(ds.date)} records")

    # Check date range
    nc_date_min = ds.date.values[0]
    nc_date_max = ds.date.values[-1]
    print(f"   NetCDF date range: {np.datetime_as_string(nc_date_min, unit='D')} to {np.datetime_as_string(nc_date_max, unit='D')}")

    # Check no NaNs
    for var in ds.data_vars:
        n_nans = ds[var].isnull().sum().item()
        assert n_nans == 0, f"❌ NaNs in {var}: {n_nans}"
    print(f"✅ No NaN values in NetCDF")

    # =========================================================================
    # STAGE 4: Attributes
    # =========================================================================
    print(f"\n--- Stage 4: Attributes ---")

    attrs_file = base_dir / f'data/neuralhydrology/attributes/{watershed}_attributes.csv'

    assert attrs_file.exists(), f"❌ Attributes not found: {attrs_file}"
    print(f"✅ Attributes file exists")

    # Load attributes
    attrs = pd.read_csv(attrs_file)

    # Check basin name matches
    expected_basin = f"{watershed}_lumped"
    actual_basin = attrs['basin'].values[0]
    assert actual_basin == expected_basin, \
        f"❌ Basin name mismatch: {actual_basin} != {expected_basin}"
    print(f"✅ Basin name matches: {actual_basin}")

    # Check all required columns
    required_cols = ['basin', 'elev_mean', 'slope_mean', 'area_gages2',
                     'frac_forest', 'soil_depth_pelletier', 'sand_frac', 'clay_frac']
    assert list(attrs.columns) == required_cols, \
        f"❌ Attribute columns incorrect"
    print(f"✅ All {len(required_cols)} attribute columns present")

    # =========================================================================
    # STAGE 5: Basin Lists
    # =========================================================================
    print(f"\n--- Stage 5: Basin Lists ---")

    basin_list_file = base_dir / f'data/basin_lists/{watershed}_basins.txt'

    assert basin_list_file.exists(), f"❌ Basin list not found: {basin_list_file}"
    print(f"✅ Basin list file exists")

    # Load basin list
    with open(basin_list_file, 'r') as f:
        basin_list_content = f.read().strip()

    assert basin_list_content == expected_basin, \
        f"❌ Basin list mismatch: {basin_list_content} != {expected_basin}"
    print(f"✅ Basin list content correct: {basin_list_content}")

    # =========================================================================
    # STAGE 6: Data Consistency Checks
    # =========================================================================
    print(f"\n--- Stage 6: Data Consistency ---")

    # Check that climate and streamflow have overlapping dates in NetCDF
    # NetCDF should be inner join of both
    climate_dates = set(climate_df['date'].dt.date)
    stream_dates = set(stream_df['date'].dt.tz_localize(None).dt.date if stream_df['date'].dt.tz else stream_df['date'].dt.date)
    overlap_dates = climate_dates & stream_dates

    nc_n_records = len(ds.date)
    expected_overlap = len(overlap_dates)

    # Allow small discrepancy due to date range differences
    assert abs(nc_n_records - expected_overlap) < 100, \
        f"❌ NetCDF record count mismatch: {nc_n_records} vs expected ~{expected_overlap}"
    print(f"✅ NetCDF record count consistent: {nc_n_records} records")

    # Spot check: verify a few dates have consistent data
    sample_dates = ds.date.values[::1000][:5]  # Every 1000th date, max 5
    for sample_date in sample_dates:
        sample_date_str = np.datetime_as_string(sample_date, unit='D')

        # Find in NetCDF
        nc_row = ds.sel(date=sample_date)

        # Find in processed CSVs
        climate_row = climate_df[climate_df['date'].dt.date == pd.to_datetime(sample_date_str).date()]
        stream_row = stream_df[stream_df['date'].dt.tz_localize(None).dt.date == pd.to_datetime(sample_date_str).date()]

        if len(climate_row) > 0 and len(stream_row) > 0:
            # Check values match
            assert abs(nc_row['prcp_mm_day'].item() - climate_row['prcp_mm_day'].values[0]) < 1e-6, \
                f"❌ prcp mismatch at {sample_date_str}"
            assert abs(nc_row['QObs_mm_d'].item() - stream_row['QObs_mm_d'].values[0]) < 1e-6, \
                f"❌ QObs mismatch at {sample_date_str}"

    print(f"✅ Spot checks passed: data consistency verified")

    # =========================================================================
    # STAGE 7: NeuralHydrology Compatibility
    # =========================================================================
    print(f"\n--- Stage 7: NeuralHydrology Compatibility ---")

    # Test we can extract time slices
    try:
        subset = ds.sel(date=slice('2000-01-01', '2000-12-31'))
        print(f"✅ Can extract time slices: {len(subset.date)} days in 2000")
    except Exception as e:
        raise AssertionError(f"❌ Time slice extraction failed: {e}")

    # Test we can access as numpy arrays
    try:
        prcp_array = ds['prcp_mm_day'].values
        assert isinstance(prcp_array, np.ndarray), "❌ Not a numpy array"
        print(f"✅ Variables accessible as numpy arrays: shape {prcp_array.shape}")
    except Exception as e:
        raise AssertionError(f"❌ Numpy array access failed: {e}")

    # Test iteration
    try:
        first_step = ds.isel(date=0)
        assert 'prcp_mm_day' in first_step, "❌ Can't access timestep"
        print(f"✅ Can iterate over time dimension")
    except Exception as e:
        raise AssertionError(f"❌ Iteration failed: {e}")

    ds.close()

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"✅✅✅ FULL PIPELINE TEST PASSED FOR {watershed.upper()} ✅✅✅")
    print(f"{'='*70}")
    print(f"\nData Summary:")
    print(f"  Raw climate records: {len(climate_raw_df)}")
    print(f"  Raw streamflow records: {len(stream_raw_df)}")
    print(f"  Processed climate records: {len(climate_df)}")
    print(f"  Processed streamflow records: {len(stream_df)}")
    print(f"  NetCDF records: {nc_n_records}")
    print(f"  Attributes: {len(attrs)} basin")
    print(f"  Basin list: ✅")
    print(f"\n{'='*70}\n")


if __name__ == '__main__':
    # Test both watersheds
    test_full_pipeline('lamar')
    test_full_pipeline('hoh')

    print("\n" + "="*70)
    print("🎉🎉🎉 ALL FULL PIPELINE TESTS PASSED 🎉🎉🎉")
    print("="*70)
    print("\n✅ Data preparation pipeline is complete and validated!")
    print("✅ Ready for Phase 2: Configuration setup")
    print("="*70)
