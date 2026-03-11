#!/usr/bin/env python3
"""
Test Stage 1.3: NetCDF Format

Verifies:
1. Variables: ['prcp_mm_day', 'tmax_C', 'tmin_C', 'QObs_mm_d']
2. Dimension: 'date' (not 'time' or other)
3. Date coordinate is datetime64
4. No NaN values in any variable
5. File can be loaded by xarray
"""

import xarray as xr
import numpy as np
from pathlib import Path


def test_netcdf_format(watershed):
    """Test NetCDF format for a watershed"""
    print(f"\n{'='*60}")
    print(f"Testing NetCDF Format: {watershed.upper()}")
    print(f"{'='*60}")

    base_dir = Path(__file__).parent.parent
    nc_file = base_dir / f'data/neuralhydrology/time_series/{watershed}_lumped.nc'

    # Test 1: File exists
    assert nc_file.exists(), f"❌ File not found: {nc_file}"
    print(f"✅ File exists: {nc_file}")

    # Test 2: File can be loaded
    try:
        ds = xr.open_dataset(nc_file)
        print(f"✅ File loaded successfully")
    except Exception as e:
        raise AssertionError(f"❌ Failed to load NetCDF: {e}")

    # Test 3: Check variables
    expected_vars = {'prcp_mm_day', 'tmax_C', 'tmin_C', 'QObs_mm_d'}
    actual_vars = set(ds.data_vars)

    assert actual_vars == expected_vars, \
        f"❌ Variable mismatch.\n  Expected: {expected_vars}\n  Got: {actual_vars}"
    print(f"✅ Variables correct: {list(ds.data_vars)}")

    # Test 4: Check dimension name
    assert 'date' in ds.dims, f"❌ 'date' dimension not found. Dims: {list(ds.dims)}"
    assert len(ds.dims) == 1, f"❌ Should have 1 dimension, got {len(ds.dims)}"
    print(f"✅ Dimension correct: 'date' with size {len(ds.date)}")

    # Test 5: Check date coordinate type
    assert ds.date.dtype == np.dtype('datetime64[ns]'), \
        f"❌ Date not datetime64[ns], got {ds.date.dtype}"
    print(f"✅ Date coordinate is datetime64[ns]")

    # Test 6: Date range and monotonicity
    date_min = ds.date.values[0]
    date_max = ds.date.values[-1]
    n_dates = len(ds.date)

    print(f"✅ Date range: {np.datetime_as_string(date_min, unit='D')} to {np.datetime_as_string(date_max, unit='D')}")
    print(f"   Records: {n_dates}")

    # Check dates are monotonically increasing
    date_diffs = np.diff(ds.date.values)
    assert np.all(date_diffs > np.timedelta64(0)), "❌ Dates not monotonically increasing"
    print(f"✅ Dates are monotonically increasing")

    # Test 7: No NaN values
    nan_counts = {}
    for var in ds.data_vars:
        n_nans = ds[var].isnull().sum().item()
        nan_counts[var] = n_nans

        if n_nans > 0:
            print(f"❌ {var}: {n_nans} NaN values")
        else:
            print(f"✅ {var}: no NaN values")

    assert all(count == 0 for count in nan_counts.values()), \
        f"❌ NaN values detected: {nan_counts}"

    # Test 8: Value ranges are reasonable
    print(f"\n=== Value Ranges ===")

    prcp_range = (ds['prcp_mm_day'].min().item(), ds['prcp_mm_day'].max().item())
    tmax_range = (ds['tmax_C'].min().item(), ds['tmax_C'].max().item())
    tmin_range = (ds['tmin_C'].min().item(), ds['tmin_C'].max().item())
    qobs_range = (ds['QObs_mm_d'].min().item(), ds['QObs_mm_d'].max().item())

    print(f"prcp_mm_day: {prcp_range[0]:.2f} to {prcp_range[1]:.2f}")
    print(f"tmax_C: {tmax_range[0]:.2f} to {tmax_range[1]:.2f}")
    print(f"tmin_C: {tmin_range[0]:.2f} to {tmin_range[1]:.2f}")
    print(f"QObs_mm_d: {qobs_range[0]:.2f} to {qobs_range[1]:.2f}")

    # Sanity checks
    assert prcp_range[0] >= 0, f"❌ Negative precipitation: {prcp_range[0]}"
    assert qobs_range[0] >= 0, f"❌ Negative streamflow: {qobs_range[0]}"
    assert tmax_range[0] > tmin_range[0], "❌ Tmax min should be > Tmin min"

    print(f"✅ All value ranges are reasonable")

    # Test 9: Check attributes exist
    print(f"\n=== Attributes ===")
    for var in ds.data_vars:
        if 'units' in ds[var].attrs:
            print(f"✅ {var} has units: {ds[var].attrs['units']}")
        else:
            print(f"⚠️  {var} missing units attribute")

    # Test 10: Verify time coverage
    # Should have data from ~1990-2025 for climate
    # But streamflow might have different coverage
    years_covered = (np.datetime_as_string(date_max, unit='Y').astype(int) -
                     np.datetime_as_string(date_min, unit='Y').astype(int))

    print(f"\n✅ Time coverage: {years_covered} years")

    # Close dataset
    ds.close()

    print(f"\n{'='*60}")
    print(f"✅ ALL TESTS PASSED for {watershed.upper()}")
    print(f"{'='*60}\n")


def test_neuralhydrology_compatibility(watershed):
    """
    Test that the NetCDF can be loaded in a way similar to NeuralHydrology.
    This is a basic compatibility check.
    """
    print(f"\n{'='*60}")
    print(f"Testing NeuralHydrology Compatibility: {watershed.upper()}")
    print(f"{'='*60}")

    base_dir = Path(__file__).parent.parent
    nc_file = base_dir / f'data/neuralhydrology/time_series/{watershed}_lumped.nc'

    # Load dataset
    ds = xr.open_dataset(nc_file)

    # Check we can extract time slices
    try:
        # Extract a subset of dates
        subset = ds.sel(date=slice('1995-01-01', '1995-12-31'))
        print(f"✅ Can extract time slices: {len(subset.date)} days in 1995")
    except Exception as e:
        raise AssertionError(f"❌ Failed to extract time slice: {e}")

    # Check we can access variables as numpy arrays
    try:
        prcp_array = ds['prcp_mm_day'].values
        assert isinstance(prcp_array, np.ndarray), "❌ Variable not convertible to numpy array"
        print(f"✅ Variables accessible as numpy arrays: shape {prcp_array.shape}")
    except Exception as e:
        raise AssertionError(f"❌ Failed to access as numpy array: {e}")

    # Check we can iterate over time dimension
    try:
        first_timestep = ds.isel(date=0)
        assert 'prcp_mm_day' in first_timestep, "❌ Can't access first timestep"
        print(f"✅ Can iterate over time dimension")
    except Exception as e:
        raise AssertionError(f"❌ Failed to iterate: {e}")

    ds.close()

    print(f"\n{'='*60}")
    print(f"✅ NeuralHydrology COMPATIBILITY TESTS PASSED for {watershed.upper()}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    # Test both watersheds
    test_netcdf_format('lamar')
    test_neuralhydrology_compatibility('lamar')

    test_netcdf_format('hoh')
    test_neuralhydrology_compatibility('hoh')

    print("\n" + "="*60)
    print("✅✅✅ ALL NETCDF FORMAT TESTS PASSED ✅✅✅")
    print("="*60)
