#!/usr/bin/env python3
"""
Run water balance model for elevation bands

This script processes climate data for multiple elevation bands (valley, mid, alpine)
and runs water balance calculations for each band with band-specific parameters.
"""

import sys
import argparse
import datetime
import json
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from water_balance import (
    WaterBalanceConfig,
    WaterBalanceResults,
    run_water_balance,
    save_results_to_csv,
    plot_water_balance
)


class ElevationBandConfig:
    """Configuration for a single elevation band"""

    def __init__(self, name: str, elevation: float, latitude: float,
                 max_soil_water: float = 100.0,
                 melt_factor: float = 0.35,
                 melt_thresh_temperature: float = -6.0,
                 precip_fraction: float = 0.167,
                 pet_type: str = 'oudin',
                 use_heatload: bool = False):
        self.name = name
        self.elevation = elevation
        self.latitude = latitude
        self.max_soil_water = max_soil_water
        self.melt_factor = melt_factor
        self.melt_thresh_temperature = melt_thresh_temperature
        self.precip_fraction = precip_fraction
        self.pet_type = pet_type
        self.use_heatload = use_heatload

    def to_wb_config(self, station: str, year_start: int, year_end: int) -> WaterBalanceConfig:
        """Convert to WaterBalanceConfig"""
        return WaterBalanceConfig(
            station=f"{station}_{self.name}",
            year_start=year_start,
            year_end=year_end,
            latitude=self.latitude,
            elevation=self.elevation,
            max_soil_water=self.max_soil_water,
            melt_factor=self.melt_factor,
            melt_thresh_temperature=self.melt_thresh_temperature,
            precip_fraction=self.precip_fraction,
            pet_type=self.pet_type,
            use_heatload=self.use_heatload,
            title=f"{station} - {self.name.capitalize()}"
        )


def load_elevation_bands_data(filepath: str) -> Dict[str, pd.DataFrame]:
    """
    Load elevation bands climate data from CSV

    Expected format:
        date,precip_valley,temp_valley,precip_mid,temp_mid,precip_alpine,temp_alpine

    Returns:
        Dictionary with keys 'valley', 'mid', 'alpine', each containing a DataFrame
    """
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])

    bands = {}

    for band in ['valley', 'mid', 'alpine']:
        band_df = pd.DataFrame({
            'date': df['date'],
            'precip': df[f'precip_{band}'],  # Already in mm
            'temp': df[f'temp_{band}']  # Already in Celsius
        })

        # For water balance, we need tmax and tmin
        # If only mean temp provided, estimate range
        # Typical diurnal range is ~10-15°C, we'll use 12°C
        diurnal_range = 12.0
        band_df['tmax'] = band_df['temp'] + (diurnal_range / 2)
        band_df['tmin'] = band_df['temp'] - (diurnal_range / 2)

        # Convert date to MM/DD/YYYY string format
        band_df['date_str'] = band_df['date'].dt.strftime('%m/%d/%Y')

        bands[band] = band_df

    return bands


def prepare_data_for_wb(band_df: pd.DataFrame) -> Dict:
    """
    Convert DataFrame to format expected by water_balance module

    GridMET data is already in mm (precip) and Celsius (temp),
    so we can pass it directly with input_units='metric'
    """
    return {
        'date': band_df['date_str'].values,
        'tmax': band_df['tmax'].values,
        'tmin': band_df['tmin'].values,
        'precip': band_df['precip'].values
    }


def load_calibrated_params(calibration_dir: str = 'output/calibration') -> Dict[str, Dict]:
    """
    Load calibrated parameters from JSON files

    Returns:
        Dictionary with band names as keys, containing calibrated parameters
    """
    calibration_path = Path(calibration_dir)
    calibrated = {}

    for band in ['valley', 'mid', 'alpine']:
        json_file = calibration_path / f'calibrated_params_{band}.json'
        if json_file.exists():
            with open(json_file, 'r') as f:
                data = json.load(f)
                calibrated[band] = data['parameters']
                print(f"  Loaded calibrated params for {band}: "
                      f"melt_factor={data['parameters']['melt_factor']:.3f}, "
                      f"melt_thresh={data['parameters']['melt_thresh_temp']:.2f}°C")

    return calibrated


def get_default_band_configs(
    station_name: str = 'Lamar',
    use_calibrated: bool = False,
    calibration_dir: str = 'output/calibration'
) -> Dict[str, ElevationBandConfig]:
    """
    Get configurations for Lamar River elevation bands

    Args:
        station_name: Name of the station/watershed
        use_calibrated: If True, load calibrated parameters from JSON files
        calibration_dir: Directory containing calibrated parameter JSON files

    You can customize these parameters based on your watershed characteristics
    """
    # Lamar River watershed approximate coordinates: 44.9°N
    latitude = 44.9

    # Load calibrated parameters if requested
    calibrated_params = {}
    if use_calibrated:
        print(f"\nLoading calibrated parameters from {calibration_dir}...")
        calibrated_params = load_calibrated_params(calibration_dir)

    # Default parameters
    default_params = {
        'valley': {'melt_factor': 0.35, 'melt_thresh_temperature': -6.0, 'precip_fraction': 0.167},
        'mid': {'melt_factor': 0.35, 'melt_thresh_temperature': -6.0, 'precip_fraction': 0.167},
        'alpine': {'melt_factor': 0.35, 'melt_thresh_temperature': -6.0, 'precip_fraction': 0.167},
    }

    # Override with calibrated parameters where available
    for band in ['valley', 'mid', 'alpine']:
        if band in calibrated_params:
            default_params[band]['melt_factor'] = calibrated_params[band]['melt_factor']
            default_params[band]['melt_thresh_temperature'] = calibrated_params[band]['melt_thresh_temp']
            default_params[band]['precip_fraction'] = calibrated_params[band]['precip_fraction']

    configs = {
        'valley': ElevationBandConfig(
            name='valley',
            elevation=2000,  # meters, approximate valley floor
            latitude=latitude,
            max_soil_water=150.0,  # mm, deeper soils in valley
            melt_factor=default_params['valley']['melt_factor'],
            melt_thresh_temperature=default_params['valley']['melt_thresh_temperature'],
            precip_fraction=default_params['valley']['precip_fraction'],
            pet_type='oudin',
            use_heatload=False
        ),
        'mid': ElevationBandConfig(
            name='mid',
            elevation=2500,  # meters, mid-elevation
            latitude=latitude,
            max_soil_water=150.0,  # mm, moderate soil depth
            melt_factor=default_params['mid']['melt_factor'],
            melt_thresh_temperature=default_params['mid']['melt_thresh_temperature'],
            precip_fraction=default_params['mid']['precip_fraction'],
            pet_type='oudin',
            use_heatload=False
        ),
        'alpine': ElevationBandConfig(
            name='alpine',
            elevation=3000,  # meters, alpine zone
            latitude=latitude,
            max_soil_water=150.0,  # mm, shallow alpine soils
            melt_factor=default_params['alpine']['melt_factor'],
            melt_thresh_temperature=default_params['alpine']['melt_thresh_temperature'],
            precip_fraction=default_params['alpine']['precip_fraction'],
            pet_type='oudin',
            use_heatload=False
        )
    }

    return configs


def run_elevation_bands_wb(
    input_file: str,
    output_dir: str,
    station_name: str = 'Lamar',
    band_configs: Dict[str, ElevationBandConfig] = None,
    create_plots: bool = True
) -> Dict[str, WaterBalanceResults]:
    """
    Run water balance for all elevation bands

    Args:
        input_file: Path to elevation bands CSV file
        output_dir: Directory for output files
        station_name: Station/watershed name
        band_configs: Dictionary of band configurations (uses defaults if None)
        create_plots: Whether to create plots

    Returns:
        Dictionary with results for each band
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading data from {input_file}...")
    bands_data = load_elevation_bands_data(input_file)

    # Get date range
    first_date = pd.to_datetime(bands_data['valley']['date'].iloc[0])
    last_date = pd.to_datetime(bands_data['valley']['date'].iloc[-1])
    year_start = first_date.year
    year_end = last_date.year

    print(f"Data spans: {first_date.date()} to {last_date.date()}")
    print(f"Years: {year_start} - {year_end}")
    print(f"Total days: {len(bands_data['valley'])}")

    # Use default configs if not provided
    if band_configs is None:
        band_configs = get_default_band_configs(station_name)

    # Run water balance for each band
    results = {}

    for band_name in ['valley', 'mid', 'alpine']:
        print(f"\n{'='*70}")
        print(f"Processing {band_name.upper()} band")
        print(f"{'='*70}")

        # Get configuration
        band_config = band_configs[band_name]
        wb_config = band_config.to_wb_config(station_name, year_start, year_end)

        print(f"  Elevation: {band_config.elevation} m")
        print(f"  Max soil water: {band_config.max_soil_water} mm")
        print(f"  Melt factor: {band_config.melt_factor}")
        print(f"  Melt threshold: {band_config.melt_thresh_temperature}°C")

        # Prepare data
        band_df = bands_data[band_name]
        wb_data = prepare_data_for_wb(band_df)

        # Run model
        print(f"  Running water balance model...")
        start_date = datetime.datetime(year_start, 1, 1)
        end_date = datetime.datetime(year_end, 12, 31)

        band_results = run_water_balance(wb_data, wb_config, start_date, end_date,
                                         input_units='metric')
        results[band_name] = band_results

        # Summary statistics
        print(f"\n  Results summary:")
        print(f"    Mean annual precip: {np.nanmean(band_results.precip) * 365:.1f} mm")
        print(f"    Mean annual PET: {np.nansum(band_results.pet) / (year_end - year_start + 1):.1f} mm")
        print(f"    Mean annual AET: {np.nansum(band_results.aet) / (year_end - year_start + 1):.1f} mm")
        print(f"    Mean annual runoff: {np.nansum(band_results.runoff) / (year_end - year_start + 1):.1f} mm")
        print(f"    Mean soil moisture: {np.nanmean(band_results.soil_water):.1f} mm")
        print(f"    Max snowpack: {np.nanmax(band_results.snowpack):.1f} mm")

        # Save results
        csv_filename = output_path / f"{station_name.lower()}_{band_name}_water_balance.csv"
        save_results_to_csv(band_results, str(csv_filename))
        print(f"    ✓ Saved CSV: {csv_filename}")

        # Create plot
        if create_plots:
            plot_filename = output_path / f"{station_name.lower()}_{band_name}_water_balance.png"
            plot_water_balance(band_results, wb_config.title, str(plot_filename))
            print(f"    ✓ Saved plot: {plot_filename}")

    # Create comparison summary
    print(f"\n{'='*70}")
    print("Creating comparison summary...")
    print(f"{'='*70}")

    comparison_file = output_path / f"{station_name.lower()}_bands_comparison.csv"
    create_comparison_summary(results, comparison_file, year_start, year_end)
    print(f"✓ Saved comparison: {comparison_file}")

    return results


def create_comparison_summary(
    results: Dict[str, WaterBalanceResults],
    output_file: Path,
    year_start: int,
    year_end: int
):
    """Create a summary comparing all elevation bands"""

    n_years = year_end - year_start + 1

    summary_data = []

    for band_name, band_results in results.items():
        summary = {
            'Band': band_name.capitalize(),
            'Mean_Annual_Precip_mm': np.nanmean(band_results.precip) * 365,
            'Mean_Annual_PET_mm': np.nansum(band_results.pet) / n_years,
            'Mean_Annual_AET_mm': np.nansum(band_results.aet) / n_years,
            'Mean_Annual_Deficit_mm': np.nansum(band_results.deficit) / n_years,
            'Mean_Annual_Runoff_mm': np.nansum(band_results.runoff) / n_years,
            'Mean_Soil_Moisture_mm': np.nanmean(band_results.soil_water),
            'Max_Snowpack_mm': np.nanmax(band_results.snowpack),
            'Mean_Peak_Snowpack_mm': np.nanmean([
                np.nanmax(band_results.snowpack[
                    (band_results.year == str(y))
                ]) for y in range(year_start, year_end + 1)
            ])
        }
        summary_data.append(summary)

    df = pd.DataFrame(summary_data)
    df.to_csv(output_file, index=False, float_format='%.2f')

    # Print to console as well
    print("\nComparison Summary:")
    print(df.to_string(index=False))


def main():
    """CLI interface"""
    parser = argparse.ArgumentParser(
        description='Run water balance for elevation bands',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--input-file',
        default='data/climate/lamar_gridmet_elevation_bands.csv',
        help='Input CSV file with elevation bands data'
    )
    parser.add_argument(
        '--output-dir',
        default='output/water_balance_bands',
        help='Output directory for results'
    )
    parser.add_argument(
        '--station-name',
        default='Lamar',
        help='Station/watershed name'
    )
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip creating plots'
    )

    # Custom band parameters
    parser.add_argument(
        '--valley-soil-capacity',
        type=float,
        default=150.0,
        help='Valley max soil water capacity (mm)'
    )
    parser.add_argument(
        '--mid-soil-capacity',
        type=float,
        default=150.0,
        help='Mid-elevation max soil water capacity (mm)'
    )
    parser.add_argument(
        '--alpine-soil-capacity',
        type=float,
        default=150.0,
        help='Alpine max soil water capacity (mm)'
    )
    parser.add_argument(
        '--valley-melt-factor',
        type=float,
        default=0.35,
        help='Valley melt factor (mm/°C/day)'
    )
    parser.add_argument(
        '--mid-melt-factor',
        type=float,
        default=0.35,
        help='Mid-elevation melt factor (mm/°C/day)'
    )
    parser.add_argument(
        '--alpine-melt-factor',
        type=float,
        default=0.35,
        help='Alpine melt factor (mm/°C/day)'
    )
    parser.add_argument(
        '--use-calibrated',
        action='store_true',
        help='Use SNOTEL-calibrated parameters from output/calibration/*.json'
    )
    parser.add_argument(
        '--calibration-dir',
        default='output/calibration',
        help='Directory containing calibrated parameter JSON files'
    )

    args = parser.parse_args()

    band_configs = get_default_band_configs(
        use_calibrated=args.use_calibrated,
        calibration_dir=args.calibration_dir
    )

    # Run water balance
    results = run_elevation_bands_wb(
        input_file=args.input_file,
        output_dir=args.output_dir,
        station_name=args.station_name,
        band_configs=band_configs,
        create_plots=not args.no_plots
    )

    print(f"\n{'='*70}")
    print("✓ All elevation bands processed successfully!")
    print(f"{'='*70}")
    print(f"\nResults saved to: {args.output_dir}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
