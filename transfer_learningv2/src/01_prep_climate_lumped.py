#!/usr/bin/env python3
"""
Stage 1.1: Lump Elevation Bands to Basin-Mean Climate

Convert elevation-band climate data to basin-mean forcing for NeuralHydrology.

Usage:
    python src/01_prep_climate_lumped.py --watershed lamar --weights 17,91,6
    python src/01_prep_climate_lumped.py --watershed hoh --weights 5,10,70
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def parse_weights(weights_str):
    """Parse weight string and normalize to sum to 1.0"""
    weights = [float(w) for w in weights_str.split(',')]
    if len(weights) != 3:
        raise ValueError("Must provide exactly 3 weights (valley, mid, alpine)")

    total = sum(weights)
    normalized = [w / total for w in weights]

    print(f"Raw weights: {weights}")
    print(f"Normalized weights: {[f'{w:.4f}' for w in normalized]}")
    print(f"Sum check: {sum(normalized):.6f}")

    return normalized


def lump_elevation_bands(input_file, output_file, weights):
    """
    Lump elevation band data to basin-mean forcing.

    Args:
        input_file: Path to elevation bands CSV
        output_file: Path to output lumped CSV
        weights: List of [valley_weight, mid_weight, alpine_weight]
    """
    print(f"\nReading: {input_file}")
    df = pd.read_csv(input_file, parse_dates=['date'])

    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Records: {len(df)}")

    # Extract weights
    w_valley, w_mid, w_alpine = weights

    # Calculate basin-mean for each variable
    # Note: GridMET precip is in mm/day, temps are in Celsius
    result = pd.DataFrame()
    result['date'] = df['date']

    # Precipitation (mm/day)
    result['prcp_mm_day'] = (
        df['precip_valley'] * w_valley +
        df['precip_mid'] * w_mid +
        df['precip_alpine'] * w_alpine
    )

    # Maximum temperature (°C)
    result['tmax_C'] = (
        df['tmax_valley'] * w_valley +
        df['tmax_mid'] * w_mid +
        df['tmax_alpine'] * w_alpine
    )

    # Minimum temperature (°C)
    result['tmin_C'] = (
        df['tmin_valley'] * w_valley +
        df['tmin_mid'] * w_mid +
        df['tmin_alpine'] * w_alpine
    )

    # Validation checks
    print("\n=== Data Validation ===")
    print(f"Missing values:")
    print(result.isnull().sum())

    print(f"\nValue ranges:")
    print(f"  prcp_mm_day: {result['prcp_mm_day'].min():.2f} to {result['prcp_mm_day'].max():.2f} mm/day")
    print(f"  tmax_C: {result['tmax_C'].min():.2f} to {result['tmax_C'].max():.2f} °C")
    print(f"  tmin_C: {result['tmin_C'].min():.2f} to {result['tmin_C'].max():.2f} °C")

    # Check for reasonable ranges
    assert result['prcp_mm_day'].min() >= 0, "Negative precipitation!"
    assert result['prcp_mm_day'].max() <= 500, f"Unreasonable precip: {result['prcp_mm_day'].max()}"
    assert result['tmax_C'].min() >= -40, f"Unreasonable tmax: {result['tmax_C'].min()}"
    assert result['tmax_C'].max() <= 50, f"Unreasonable tmax: {result['tmax_C'].max()}"
    assert result['tmin_C'].min() >= -50, f"Unreasonable tmin: {result['tmin_C'].min()}"
    assert result['tmin_C'].max() <= 40, f"Unreasonable tmin: {result['tmin_C'].max()}"
    assert result.isnull().sum().sum() == 0, "Missing values detected!"

    print("\n✅ All validation checks passed")

    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_file, index=False)
    print(f"\nSaved: {output_file}")
    print(f"Output shape: {result.shape}")
    print(f"Columns: {list(result.columns)}")

    return result


def main():
    parser = argparse.ArgumentParser(description='Lump elevation bands to basin-mean climate')
    parser.add_argument('--watershed', required=True, choices=['lamar', 'hoh'],
                        help='Watershed name')
    parser.add_argument('--weights', required=True,
                        help='Elevation weights as "valley,mid,alpine" (e.g., "17,91,6")')

    args = parser.parse_args()

    # Setup paths
    base_dir = Path(__file__).parent.parent
    input_file = base_dir / f'data/raw/{args.watershed}_gridmet_elevation_bands.csv'
    output_file = base_dir / f'data/processed/{args.watershed}_lumped.csv'

    # Parse and validate weights
    weights = parse_weights(args.weights)

    # Process
    lump_elevation_bands(input_file, output_file, weights)


if __name__ == '__main__':
    main()
