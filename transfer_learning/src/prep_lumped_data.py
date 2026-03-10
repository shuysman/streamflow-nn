#!/usr/bin/env python3
"""
Prepare lumped watershed data for NeuralHydrology transfer learning.

This script aggregates elevation-band climate data into basin-mean forcing
and combines it with streamflow observations to create a NetCDF file
compatible with NeuralHydrology's generic dataset format.

Usage:
    python prep_lumped_data.py --watershed lamar
    python prep_lumped_data.py --watershed hoh --weights 10,85,5
    python prep_lumped_data.py --watershed lamar --weights-file lamar_weights.json

Author: Transfer Learning Pipeline
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def load_elevation_weights(watershed: str, weights_arg: str = None, weights_file: str = None) -> dict:
    """
    Load or calculate elevation band weights.

    Priority:
    1. Explicit --weights argument (comma-separated: valley,mid,alpine)
    2. Weights JSON file
    3. Default equal weights (1/3 each)

    Returns:
        dict with 'valley', 'mid', 'alpine' weights summing to 1.0
    """
    # Option 1: Explicit weights argument
    if weights_arg:
        parts = [float(x.strip()) for x in weights_arg.split(',')]
        if len(parts) != 3:
            raise ValueError("--weights must have 3 values: valley,mid,alpine")
        total = sum(parts)
        return {
            'valley': parts[0] / total,
            'mid': parts[1] / total,
            'alpine': parts[2] / total
        }

    # Option 2: Weights file
    if weights_file and Path(weights_file).exists():
        with open(weights_file, 'r') as f:
            data = json.load(f)
        # Normalize
        total = data['valley'] + data['mid'] + data['alpine']
        return {k: v / total for k, v in data.items()}

    # Option 3: Check for default weights file location
    default_weights_path = Path(f'transfer_learning/data/climate/{watershed}_elevation_weights.json')
    if default_weights_path.exists():
        with open(default_weights_path, 'r') as f:
            data = json.load(f)
        total = data['valley'] + data['mid'] + data['alpine']
        print(f"  Using weights from {default_weights_path}")
        return {k: v / total for k, v in data.items()}

    # Option 4: Known watershed defaults
    known_weights = {
        'lamar': {'valley': 17, 'mid': 91, 'alpine': 6},  # From download_gridmet_elevation_bands.py
        'hoh': {'valley': 5, 'mid': 10, 'alpine': 70},    # From download_hoh_gridmet_elevation_bands.py
    }

    if watershed in known_weights:
        data = known_weights[watershed]
        total = sum(data.values())
        print(f"  Using known weights for {watershed}: {data}")
        return {k: v / total for k, v in data.items()}

    # Fallback: Equal weights
    print(f"  Warning: No weights found for {watershed}, using equal weights (1/3 each)")
    return {'valley': 1/3, 'mid': 1/3, 'alpine': 1/3}


def find_data_files(watershed: str, data_dir: str = 'transfer_learning/data') -> tuple:
    """
    Locate streamflow and climate data files for a watershed.

    Returns:
        tuple: (streamflow_path, climate_path)
    """
    data_path = Path(data_dir)

    # Try common naming patterns for streamflow
    streamflow_patterns = [
        f'{watershed}_river_streamflow.csv',
        f'{watershed}_streamflow.csv',
        f'{watershed}.csv',
    ]

    streamflow_path = None
    for pattern in streamflow_patterns:
        candidate = data_path / pattern
        if candidate.exists():
            streamflow_path = candidate
            break

    if not streamflow_path:
        raise FileNotFoundError(
            f"Could not find streamflow data for '{watershed}'. "
            f"Tried: {', '.join(streamflow_patterns)} in {data_path}"
        )

    # Try common naming patterns for climate
    climate_patterns = [
        f'climate/{watershed}_gridmet_elevation_bands.csv',
        f'climate/{watershed}_elevation_bands.csv',
        f'climate/{watershed}_climate.csv',
    ]

    climate_path = None
    for pattern in climate_patterns:
        candidate = data_path / pattern
        if candidate.exists():
            climate_path = candidate
            break

    if not climate_path:
        raise FileNotFoundError(
            f"Could not find climate data for '{watershed}'. "
            f"Tried: {', '.join(climate_patterns)} in {data_path}"
        )

    return streamflow_path, climate_path


def prep_lumped_data(
    watershed: str,
    weights: dict,
    streamflow_path: Path,
    climate_path: Path,
    output_dir: str = 'transfer_learning/data/neuralhydrology/time_series',
    streamflow_col: str = 'streamflow_cfs',
    three_var_only: bool = False
) -> Path:
    """
    Prepare lumped watershed data for NeuralHydrology.

    Args:
        watershed: Watershed name (used for output filename)
        weights: Dict with 'valley', 'mid', 'alpine' weights
        streamflow_path: Path to streamflow CSV
        climate_path: Path to elevation-band climate CSV
        output_dir: Output directory for NetCDF
        streamflow_col: Column name for streamflow in input CSV

    Returns:
        Path to output NetCDF file
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading streamflow from {streamflow_path}...")
    streamflow = pd.read_csv(streamflow_path, parse_dates=['date']).set_index('date')

    print(f"Loading climate from {climate_path}...")
    climate = pd.read_csv(climate_path, parse_dates=['date']).set_index('date')

    # Handle timezone-aware indices
    if streamflow.index.tz is not None:
        streamflow.index = streamflow.index.tz_localize(None)
    if climate.index.tz is not None:
        climate.index = climate.index.tz_localize(None)

    print(f"  Streamflow records: {len(streamflow)}")
    print(f"  Climate records: {len(climate)}")
    print(f"  Weights: valley={weights['valley']:.3f}, mid={weights['mid']:.3f}, alpine={weights['alpine']:.3f}")

    # Compute weighted averages
    df = pd.DataFrame(index=climate.index)

    # Precipitation
    df['prcp'] = (
        climate['precip_valley'] * weights['valley'] +
        climate['precip_mid'] * weights['mid'] +
        climate['precip_alpine'] * weights['alpine']
    ).astype('float32')

    # Max temperature
    df['tmax'] = (
        climate['tmax_valley'] * weights['valley'] +
        climate['tmax_mid'] * weights['mid'] +
        climate['tmax_alpine'] * weights['alpine']
    ).astype('float32')

    # Min temperature
    df['tmin'] = (
        climate['tmin_valley'] * weights['valley'] +
        climate['tmin_mid'] * weights['mid'] +
        climate['tmin_alpine'] * weights['alpine']
    ).astype('float32')

    # Solar radiation (if available)
    if 'srad_valley' in climate.columns:
        df['srad'] = (
            climate['srad_valley'] * weights['valley'] +
            climate['srad_mid'] * weights['mid'] +
            climate['srad_alpine'] * weights['alpine']
        ).astype('float32')
        print(f"  Added srad (solar radiation)")

    # Vapor pressure (if available)
    if 'vp_valley' in climate.columns:
        df['vp'] = (
            climate['vp_valley'] * weights['valley'] +
            climate['vp_mid'] * weights['mid'] +
            climate['vp_alpine'] * weights['alpine']
        ).astype('float32')
        print(f"  Added vp (vapor pressure)")

    # Join streamflow
    if streamflow_col not in streamflow.columns:
        available = ', '.join(streamflow.columns.tolist())
        raise ValueError(f"Column '{streamflow_col}' not found. Available: {available}")

    df = df.join(streamflow[[streamflow_col]], how='inner')
    df[streamflow_col] = df[streamflow_col].astype('float32')

    print(f"  Joined records: {len(df)}")
    print(f"  Date range: {df.index.min()} to {df.index.max()}")

    # Check for missing data
    missing = df.isnull().sum()
    if missing.any():
        print(f"  Warning: Missing values detected:")
        for col, count in missing[missing > 0].items():
            print(f"    {col}: {count} missing")

    # Create xarray Dataset
    data_vars = {
        'prcp': (['date'], df['prcp'].values),
        'tmax': (['date'], df['tmax'].values),
        'tmin': (['date'], df['tmin'].values),
        streamflow_col: (['date'], df[streamflow_col].values),
    }

    # Add optional variables if present
    if 'srad' in df.columns:
        data_vars['srad'] = (['date'], df['srad'].values)
    if 'vp' in df.columns:
        data_vars['vp'] = (['date'], df['vp'].values)

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            'date': df.index.values
        }
    )

    # Add metadata
    ds.attrs['watershed'] = watershed
    ds.attrs['weights_valley'] = weights['valley']
    ds.attrs['weights_mid'] = weights['mid']
    ds.attrs['weights_alpine'] = weights['alpine']

    # Save
    output_path = Path(output_dir) / f'{watershed}_lumped.nc'
    ds.to_netcdf(output_path)

    print(f"\nSaved: {output_path}")
    print(f"  Variables: {list(ds.data_vars)}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Prepare lumped watershed data for NeuralHydrology transfer learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --watershed lamar
  %(prog)s --watershed hoh --weights 10,85,5
  %(prog)s --watershed lamar --streamflow-path data/custom_flow.csv
        """
    )

    parser.add_argument(
        '--watershed', '-w',
        required=True,
        help="Watershed name (e.g., 'lamar', 'hoh')"
    )
    parser.add_argument(
        '--weights',
        help="Elevation band weights as comma-separated values: valley,mid,alpine (e.g., '17,91,6')"
    )
    parser.add_argument(
        '--weights-file',
        help="Path to JSON file with elevation weights"
    )
    parser.add_argument(
        '--data-dir',
        default='transfer_learning/data',
        help="Base data directory (default: transfer_learning/data)"
    )
    parser.add_argument(
        '--streamflow-path',
        help="Override path to streamflow CSV"
    )
    parser.add_argument(
        '--climate-path',
        help="Override path to climate CSV"
    )
    parser.add_argument(
        '--output-dir',
        default='transfer_learning/data/neuralhydrology/time_series',
        help="Output directory for NetCDF"
    )
    parser.add_argument(
        '--streamflow-col',
        default='streamflow_cfs',
        help="Column name for streamflow (default: streamflow_cfs)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print(f"Preparing Lumped Data for: {args.watershed}")
    print("=" * 60)

    # Load weights
    print("\nLoading elevation weights...")
    weights = load_elevation_weights(
        args.watershed,
        weights_arg=args.weights,
        weights_file=args.weights_file
    )

    # Find data files
    print("\nLocating data files...")
    if args.streamflow_path and args.climate_path:
        streamflow_path = Path(args.streamflow_path)
        climate_path = Path(args.climate_path)
    else:
        streamflow_path, climate_path = find_data_files(args.watershed, args.data_dir)

    print(f"  Streamflow: {streamflow_path}")
    print(f"  Climate: {climate_path}")

    # Prepare data
    print("\nProcessing...")
    output_path = prep_lumped_data(
        watershed=args.watershed,
        weights=weights,
        streamflow_path=streamflow_path,
        climate_path=climate_path,
        output_dir=args.output_dir,
        streamflow_col=args.streamflow_col
    )

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)

    return output_path


if __name__ == "__main__":
    main()
