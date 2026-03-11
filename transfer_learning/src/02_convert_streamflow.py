#!/usr/bin/env python3
"""
Stage 1.2: Convert Streamflow Units (CFS → mm/day)

Convert streamflow from cubic feet per second (CFS) to millimeters per day (mm/day)
for NeuralHydrology compatibility.

Formula:
    1. CFS to cubic meters per second: cms = cfs * 0.0283168
    2. Cubic meters per day: cmd = cms * 86400
    3. Depth over watershed: mm_day = (cmd / (area_km2 * 1e6)) * 1000

Usage:
    python src/02_convert_streamflow.py --watershed lamar --area 1730.1
    python src/02_convert_streamflow.py --watershed hoh --area 655.3
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path


# Conversion constants
CFS_TO_CMS = 0.0283168  # cubic feet/sec to cubic meters/sec
SECONDS_PER_DAY = 86400


def cfs_to_mmday(cfs, drainage_area_km2):
    """
    Convert streamflow from CFS to mm/day.

    Args:
        cfs: Streamflow in cubic feet per second
        drainage_area_km2: Drainage area in square kilometers

    Returns:
        Streamflow in millimeters per day
    """
    # CFS to cubic meters per second
    cms = cfs * CFS_TO_CMS

    # Cubic meters per day
    cmd = cms * SECONDS_PER_DAY

    # Depth over watershed (mm/day)
    # cmd / (area_m2) = depth in meters
    # Convert to mm by multiplying by 1000
    area_m2 = drainage_area_km2 * 1e6
    mm_day = (cmd / area_m2) * 1000

    return mm_day


def mmday_to_cfs(mm_day, drainage_area_km2):
    """
    Reverse conversion: mm/day to CFS (for validation).

    Args:
        mm_day: Streamflow in millimeters per day
        drainage_area_km2: Drainage area in square kilometers

    Returns:
        Streamflow in cubic feet per second
    """
    area_m2 = drainage_area_km2 * 1e6

    # mm/day to meters/day
    m_day = mm_day / 1000

    # Volume per day (cubic meters)
    cmd = m_day * area_m2

    # Cubic meters per second
    cms = cmd / SECONDS_PER_DAY

    # CFS
    cfs = cms / CFS_TO_CMS

    return cfs


def convert_streamflow(input_file, output_file, drainage_area_km2):
    """
    Convert streamflow from CFS to mm/day.

    Args:
        input_file: Path to input CSV (with streamflow_cfs column)
        output_file: Path to output CSV (with QObs_mm_d column)
        drainage_area_km2: Drainage area in square kilometers
    """
    print(f"\nReading: {input_file}")
    df = pd.read_csv(input_file, parse_dates=['date'])

    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Records: {len(df)}")

    # Convert CFS to mm/day
    print(f"\nDrainage area: {drainage_area_km2} km²")
    df['QObs_mm_d'] = cfs_to_mmday(df['streamflow_cfs'], drainage_area_km2)

    # Validation checks
    print("\n=== Data Validation ===")

    # Check for negative values
    n_negative = (df['QObs_mm_d'] < 0).sum()
    if n_negative > 0:
        print(f"⚠️  Warning: {n_negative} negative values (setting to 0)")
        df.loc[df['QObs_mm_d'] < 0, 'QObs_mm_d'] = 0

    # Check for missing values
    missing = df[['date', 'QObs_mm_d']].isnull().sum()
    print(f"Missing values:")
    print(missing)
    assert missing.sum() == 0, "Missing values detected!"

    # Display range
    print(f"\nValue ranges:")
    print(f"  Original (CFS): {df['streamflow_cfs'].min():.2f} to {df['streamflow_cfs'].max():.2f}")
    print(f"  Converted (mm/day): {df['QObs_mm_d'].min():.2f} to {df['QObs_mm_d'].max():.2f}")
    print(f"  Mean: {df['QObs_mm_d'].mean():.2f} mm/day")
    print(f"  Median: {df['QObs_mm_d'].median():.2f} mm/day")

    # Sanity check: reasonable range for streamflow
    assert df['QObs_mm_d'].min() >= 0, "Negative streamflow!"
    assert df['QObs_mm_d'].max() < 500, f"Unreasonably high flow: {df['QObs_mm_d'].max()}"

    # Test reversibility (round-trip conversion)
    print("\n=== Reversibility Test ===")
    cfs_roundtrip = mmday_to_cfs(df['QObs_mm_d'], drainage_area_km2)
    max_error = np.abs(cfs_roundtrip - df['streamflow_cfs']).max()
    mean_error = np.abs(cfs_roundtrip - df['streamflow_cfs']).mean()

    print(f"Round-trip error (CFS):")
    print(f"  Max: {max_error:.6f}")
    print(f"  Mean: {mean_error:.6f}")

    assert max_error < 1e-6, f"Round-trip conversion failed! Max error: {max_error}"
    print("✅ Conversion is reversible")

    print("\n✅ All validation checks passed")

    # Save only date and QObs_mm_d
    output_df = df[['date', 'QObs_mm_d']].copy()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_file, index=False)

    print(f"\nSaved: {output_file}")
    print(f"Output shape: {output_df.shape}")
    print(f"Columns: {list(output_df.columns)}")

    return output_df


def main():
    parser = argparse.ArgumentParser(description='Convert streamflow from CFS to mm/day')
    parser.add_argument('--watershed', required=True, choices=['lamar', 'hoh'],
                        help='Watershed name')
    parser.add_argument('--area', required=True, type=float,
                        help='Drainage area in square kilometers')

    args = parser.parse_args()

    # Setup paths
    base_dir = Path(__file__).parent.parent
    input_file = base_dir / f'data/raw/{args.watershed}_streamflow.csv'
    output_file = base_dir / f'data/processed/{args.watershed}_streamflow_mmday.csv'

    # Process
    convert_streamflow(input_file, output_file, args.area)


if __name__ == '__main__':
    main()
