#!/usr/bin/env python3
"""
Stage 1.4: Calculate Static Attributes

Generate CAMELS-style static attributes for each watershed.
These are used by NeuralHydrology to provide basin characteristics to the model.

Attributes (from plan):
- basin: Basin identifier (must match NetCDF filename without .nc)
- elev_mean: Mean elevation (m)
- slope_mean: Mean slope (degrees or %)
- area_gages2: Drainage area (km²)
- frac_forest: Forest fraction (0-1)
- soil_depth_pelletier: Soil depth (m)
- sand_frac: Sand fraction (0-1)
- clay_frac: Clay fraction (0-1)

Usage:
    python src/04_calculate_attributes.py --watershed lamar
    python src/04_calculate_attributes.py --watershed hoh
"""

import argparse
import pandas as pd
from pathlib import Path


# Watershed attributes (from plan and watershed characteristics)
WATERSHED_ATTRIBUTES = {
    'lamar': {
        'basin': 'lamar_lumped',
        'elev_mean': 2584.4,      # Mean elevation in meters
        'slope_mean': 4.9,         # Mean slope (degrees)
        'area_gages2': 1730.1,     # Drainage area in km²
        'frac_forest': 0.33,       # Forest fraction (33%)
        'soil_depth_pelletier': 1.2,   # Soil depth in meters
        'sand_frac': 0.4,          # Sand fraction
        'clay_frac': 0.2           # Clay fraction
    },
    'hoh': {
        'basin': 'hoh_lumped',
        'elev_mean': 706.1,        # Mean elevation in meters
        'slope_mean': 25.0,        # Mean slope (degrees) - steep terrain
        'area_gages2': 655.3,      # Drainage area in km²
        'frac_forest': 0.85,       # Forest fraction (85% - temperate rainforest)
        'soil_depth_pelletier': 1.15,  # Soil depth in meters
        'sand_frac': 0.35,         # Sand fraction
        'clay_frac': 0.25          # Clay fraction
    }
}


def calculate_attributes(watershed, output_file):
    """
    Create static attributes CSV for a watershed.

    Args:
        watershed: Watershed name ('lamar' or 'hoh')
        output_file: Path to output CSV file
    """
    print(f"\n{'='*60}")
    print(f"Calculating Attributes: {watershed.upper()}")
    print(f"{'='*60}")

    if watershed not in WATERSHED_ATTRIBUTES:
        raise ValueError(f"Unknown watershed: {watershed}. Must be 'lamar' or 'hoh'")

    # Get attributes
    attrs = WATERSHED_ATTRIBUTES[watershed]

    print(f"\nBasin identifier: {attrs['basin']}")
    print(f"\nPhysical characteristics:")
    print(f"  Mean elevation: {attrs['elev_mean']} m")
    print(f"  Mean slope: {attrs['slope_mean']} degrees")
    print(f"  Drainage area: {attrs['area_gages2']} km²")
    print(f"  Forest fraction: {attrs['frac_forest']*100:.1f}%")
    print(f"  Soil depth: {attrs['soil_depth_pelletier']} m")
    print(f"  Sand fraction: {attrs['sand_frac']*100:.1f}%")
    print(f"  Clay fraction: {attrs['clay_frac']*100:.1f}%")

    # Create DataFrame (single row)
    df = pd.DataFrame([attrs])

    # Validation
    print(f"\n=== Validation ===")

    # Check basin name format
    expected_basin_name = f"{watershed}_lumped"
    assert df['basin'].values[0] == expected_basin_name, \
        f"Basin name mismatch: {df['basin'].values[0]} != {expected_basin_name}"
    print(f"✅ Basin name matches expected: {expected_basin_name}")

    # Check all required columns present
    required_cols = ['basin', 'elev_mean', 'slope_mean', 'area_gages2',
                     'frac_forest', 'soil_depth_pelletier', 'sand_frac', 'clay_frac']
    assert list(df.columns) == required_cols, \
        f"Column mismatch. Got: {list(df.columns)}"
    print(f"✅ All required columns present: {len(required_cols)} columns")

    # Check all values are numeric (except basin)
    for col in required_cols[1:]:  # Skip 'basin'
        assert pd.api.types.is_numeric_dtype(df[col]), \
            f"Column {col} is not numeric: {df[col].dtype}"
    print(f"✅ All attribute values are numeric")

    # Check reasonable ranges
    assert 0 < df['elev_mean'].values[0] < 5000, "Elevation out of range"
    assert 0 <= df['slope_mean'].values[0] < 90, "Slope out of range"
    assert df['area_gages2'].values[0] > 0, "Drainage area must be positive"
    assert 0 <= df['frac_forest'].values[0] <= 1, "Forest fraction must be 0-1"
    assert df['soil_depth_pelletier'].values[0] > 0, "Soil depth must be positive"
    assert 0 <= df['sand_frac'].values[0] <= 1, "Sand fraction must be 0-1"
    assert 0 <= df['clay_frac'].values[0] <= 1, "Clay fraction must be 0-1"
    print(f"✅ All values in reasonable ranges")

    # Check fractions sum to reasonable value (sand + clay should be < 1, leaving room for silt)
    frac_sum = df['sand_frac'].values[0] + df['clay_frac'].values[0]
    assert frac_sum <= 1.0, f"Sand + clay fractions exceed 1.0: {frac_sum}"
    print(f"✅ Sand + clay fractions sum to {frac_sum:.2f} (leaving room for silt)")

    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)

    print(f"\n✅ Saved: {output_file}")
    print(f"   Shape: {df.shape}")

    # Display CSV content
    print(f"\n=== CSV Content ===")
    print(df.to_string(index=False))

    return df


def main():
    parser = argparse.ArgumentParser(description='Calculate static attributes for watershed')
    parser.add_argument('--watershed', required=True, choices=['lamar', 'hoh'],
                        help='Watershed name')

    args = parser.parse_args()

    # Setup paths
    base_dir = Path(__file__).parent.parent
    output_file = base_dir / f'data/neuralhydrology/attributes/{args.watershed}_attributes.csv'

    # Process
    calculate_attributes(args.watershed, output_file)


if __name__ == '__main__':
    main()
