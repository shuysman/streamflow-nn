#!/usr/bin/env python3
"""
Test Stage 1.2: Streamflow Conversion

Verifies:
1. Column name is exactly 'QObs_mm_d'
2. Date range matches raw data
3. No negative values
4. Reasonable range: 0.1-500 mm/day
5. Conversion is reversible
"""

import pandas as pd
import numpy as np
from pathlib import Path


# Known drainage areas
DRAINAGE_AREAS = {
    'lamar': 1730.1,  # km²
    'hoh': 655.3      # km²
}


def cfs_to_mmday(cfs, area_km2):
    """Convert CFS to mm/day"""
    cms = cfs * 0.0283168
    cmd = cms * 86400
    mm_day = (cmd / (area_km2 * 1e6)) * 1000
    return mm_day


def mmday_to_cfs(mm_day, area_km2):
    """Convert mm/day to CFS"""
    area_m2 = area_km2 * 1e6
    cmd = (mm_day / 1000) * area_m2
    cms = cmd / 86400
    cfs = cms / 0.0283168
    return cfs


def test_streamflow_conversion(watershed):
    """Test streamflow conversion for a watershed"""
    print(f"\n{'='*60}")
    print(f"Testing Streamflow Conversion: {watershed.upper()}")
    print(f"{'='*60}")

    base_dir = Path(__file__).parent.parent

    # Load raw data
    raw_file = base_dir / f'data/raw/{watershed}_streamflow.csv'
    assert raw_file.exists(), f"❌ Raw file not found: {raw_file}"
    raw_df = pd.read_csv(raw_file, parse_dates=['date'])
    print(f"✅ Raw data loaded: {len(raw_df)} records")

    # Load converted data
    converted_file = base_dir / f'data/processed/{watershed}_streamflow_mmday.csv'
    assert converted_file.exists(), f"❌ Converted file not found: {converted_file}"
    print(f"✅ File exists: {converted_file}")

    df = pd.read_csv(converted_file, parse_dates=['date'])
    print(f"✅ File loaded successfully")
    print(f"   Shape: {df.shape}")

    # Test 1: Column names
    expected_cols = ['date', 'QObs_mm_d']
    assert list(df.columns) == expected_cols, f"❌ Column mismatch. Got: {list(df.columns)}"
    print(f"✅ Column names correct: {expected_cols}")

    # Test 2: Date range matches raw data
    assert df['date'].min() == raw_df['date'].min(), "❌ Start date mismatch"
    assert df['date'].max() == raw_df['date'].max(), "❌ End date mismatch"
    assert len(df) == len(raw_df), "❌ Record count mismatch"
    print(f"✅ Date range matches raw data: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"   Records: {len(df)}")

    # Test 3: No negative values
    n_negative = (df['QObs_mm_d'] < 0).sum()
    assert n_negative == 0, f"❌ {n_negative} negative values found"
    print(f"✅ No negative values")

    # Test 4: Reasonable range
    min_flow = df['QObs_mm_d'].min()
    max_flow = df['QObs_mm_d'].max()
    mean_flow = df['QObs_mm_d'].mean()
    median_flow = df['QObs_mm_d'].median()

    print(f"✅ Value ranges:")
    print(f"   Min: {min_flow:.2f} mm/day")
    print(f"   Max: {max_flow:.2f} mm/day")
    print(f"   Mean: {mean_flow:.2f} mm/day")
    print(f"   Median: {median_flow:.2f} mm/day")

    # Check reasonable bounds
    assert min_flow >= 0, f"❌ Negative minimum: {min_flow}"
    assert max_flow < 500, f"❌ Unreasonably high max: {max_flow}"
    assert mean_flow > 0.1, f"❌ Mean too low: {mean_flow}"

    # Test 5: Conversion is reversible
    print("\n=== Reversibility Test ===")
    area_km2 = DRAINAGE_AREAS[watershed]

    # Convert back to CFS
    cfs_roundtrip = mmday_to_cfs(df['QObs_mm_d'].values, area_km2)

    # Compare with original
    max_error = np.abs(cfs_roundtrip - raw_df['streamflow_cfs'].values).max()
    mean_error = np.abs(cfs_roundtrip - raw_df['streamflow_cfs'].values).mean()
    rel_error = (np.abs(cfs_roundtrip - raw_df['streamflow_cfs'].values) /
                 (raw_df['streamflow_cfs'].values + 1e-10)).max()

    print(f"Round-trip error:")
    print(f"   Max absolute: {max_error:.6f} CFS")
    print(f"   Mean absolute: {mean_error:.6f} CFS")
    print(f"   Max relative: {rel_error*100:.6f}%")

    assert max_error < 1e-6, f"❌ Round-trip failed: max error {max_error}"
    assert np.allclose(cfs_roundtrip, raw_df['streamflow_cfs'].values, rtol=1e-9), \
        "❌ Round-trip conversion not close enough"

    print(f"✅ Conversion is reversible (error < 1e-6)")

    # Test 6: Spot check a few conversions manually
    print("\n=== Spot Checks ===")
    for idx in [0, len(df)//2, -1]:
        original_cfs = raw_df.iloc[idx]['streamflow_cfs']
        converted_mmday = df.iloc[idx]['QObs_mm_d']
        expected_mmday = cfs_to_mmday(original_cfs, area_km2)

        error = abs(converted_mmday - expected_mmday)
        print(f"   Row {idx}: {original_cfs:.2f} CFS → {converted_mmday:.2f} mm/day (error: {error:.9f})")
        assert error < 1e-9, f"❌ Spot check failed at row {idx}"

    print(f"✅ Spot checks passed")

    print(f"\n{'='*60}")
    print(f"✅ ALL TESTS PASSED for {watershed.upper()}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    # Test both watersheds
    test_streamflow_conversion('lamar')
    test_streamflow_conversion('hoh')

    print("\n" + "="*60)
    print("✅✅✅ ALL STREAMFLOW CONVERSION TESTS PASSED ✅✅✅")
    print("="*60)
