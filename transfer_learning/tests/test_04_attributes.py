#!/usr/bin/env python3
"""
Test Stage 1.4: Static Attributes

Verifies:
1. Basin name matches NetCDF filename (without .nc)
2. All 7 attributes present
3. area_gages2 matches USGS gage values
4. Values are float (not strings)
5. Values are in reasonable ranges
"""

import pandas as pd
from pathlib import Path


# USGS gage drainage areas (authoritative)
USGS_AREAS = {
    'lamar': 1730.1,
    'hoh': 655.3
}


def test_attributes(watershed):
    """Test attributes for a watershed"""
    print(f"\n{'='*60}")
    print(f"Testing Attributes: {watershed.upper()}")
    print(f"{'='*60}")

    base_dir = Path(__file__).parent.parent

    # Test 1: File exists
    attrs_file = base_dir / f'data/neuralhydrology/attributes/{watershed}_attributes.csv'
    assert attrs_file.exists(), f"File not found: {attrs_file}"
    print(f"  File exists: {attrs_file}")

    # Test 2: Load file
    df = pd.read_csv(attrs_file)
    print(f"  File loaded successfully, shape: {df.shape}")

    # Test 3: Check it's a single row
    assert len(df) == 1, f"Should have exactly 1 row, got {len(df)}"
    print(f"  Single row (one basin)")

    # Test 4: Check basin name matches expected format
    expected_basin = f"{watershed}_lumped"
    actual_basin = df['basin'].values[0]
    assert actual_basin == expected_basin, \
        f"Basin name mismatch: {actual_basin} != {expected_basin}"
    print(f"  Basin name correct: {actual_basin}")

    # Test 5: Check NetCDF file exists with matching name
    nc_file = base_dir / f'data/neuralhydrology/time_series/{watershed}_lumped.nc'
    assert nc_file.exists(), f"Corresponding NetCDF not found: {nc_file}"
    print(f"  Corresponding NetCDF exists")

    # Test 6: Check all required columns
    required_cols = ['basin', 'elev_mean', 'slope_mean', 'area_gages2',
                     'frac_forest', 'soil_depth_pelletier', 'sand_frac', 'clay_frac']
    actual_cols = list(df.columns)

    assert actual_cols == required_cols, \
        f"Column mismatch.\n  Expected: {required_cols}\n  Got: {actual_cols}"
    print(f"  All {len(required_cols)} required columns present")

    # Test 7: Check all numeric columns are float
    numeric_cols = required_cols[1:]  # Skip 'basin'
    for col in numeric_cols:
        assert pd.api.types.is_numeric_dtype(df[col]), \
            f"Column {col} not numeric: {df[col].dtype}"
    print(f"  All numeric columns have numeric dtype")

    # Test 8: Check area_gages2 matches USGS gage value
    area_from_attrs = df['area_gages2'].values[0]
    expected_area = USGS_AREAS[watershed]
    assert abs(area_from_attrs - expected_area) < 0.1, \
        f"Drainage area mismatch: {area_from_attrs} != {expected_area}"
    print(f"  Drainage area matches USGS: {area_from_attrs} km2")

    # Test 9: Check value ranges
    print(f"\n  === Values ===")

    elev = df['elev_mean'].values[0]
    slope = df['slope_mean'].values[0]
    area = df['area_gages2'].values[0]
    forest = df['frac_forest'].values[0]
    soil = df['soil_depth_pelletier'].values[0]
    sand = df['sand_frac'].values[0]
    clay = df['clay_frac'].values[0]

    print(f"    elev_mean: {elev} m")
    print(f"    slope_mean: {slope} degrees")
    print(f"    area_gages2: {area} km2")
    print(f"    frac_forest: {forest*100:.1f}%")
    print(f"    soil_depth_pelletier: {soil} m")
    print(f"    sand_frac: {sand*100:.1f}%")
    print(f"    clay_frac: {clay*100:.1f}%")

    # Reasonable range checks
    assert 0 < elev < 5000, f"Elevation out of range: {elev}"
    assert 0 < slope < 60, f"Slope out of range: {slope}"
    assert area > 0, f"Area must be positive: {area}"
    assert 0 <= forest <= 1, f"Forest fraction out of range: {forest}"
    assert 0 < soil < 10, f"Soil depth out of range: {soil}"
    assert 0 <= sand <= 1, f"Sand fraction out of range: {sand}"
    assert 0 <= clay <= 1, f"Clay fraction out of range: {clay}"

    print(f"  All values in reasonable ranges")

    # Test 10: Soil texture consistency
    frac_sum = sand + clay
    assert frac_sum <= 1.0, f"Sand + clay exceeds 1.0: {frac_sum}"
    silt_frac = 1.0 - frac_sum
    print(f"  Soil texture consistent (implied silt: {silt_frac*100:.1f}%)")

    # Test 11: Watershed-specific checks
    print(f"\n  === Watershed-Specific Checks ===")

    if watershed == 'lamar':
        # Lamar is high elevation, mountainous, moderate forest
        assert elev > 2000, f"Lamar should be high elevation: {elev}"
        assert 5 < slope < 30, f"Lamar slope should be 5-30 degrees: {slope}"
        assert forest < 0.6, f"Lamar has limited forest: {forest}"
        print(f"  Lamar characteristics match expectations")
    elif watershed == 'hoh':
        # Hoh is lower elevation, steep, dense forest
        assert elev < 1500, f"Hoh should be lower elevation: {elev}"
        assert slope > 10, f"Hoh should be steep: {slope}"
        assert forest > 0.5, f"Hoh is temperate rainforest: {forest}"
        print(f"  Hoh characteristics match expectations")

    print(f"\n{'='*60}")
    print(f"ALL TESTS PASSED for {watershed.upper()}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    # Test both watersheds
    test_attributes('lamar')
    test_attributes('hoh')

    print("\n" + "="*60)
    print("ALL ATTRIBUTES TESTS PASSED")
    print("="*60)
