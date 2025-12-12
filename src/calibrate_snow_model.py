#!/usr/bin/env python3
"""
SNOTEL-Based Calibration of Water Balance Snow Model

This script calibrates the degree-day snow model parameters using observed
SWE from SNOTEL stations.

SNOTEL Station → Elevation Band Mapping:
- Parker Peak (683): 9420 ft (2871m) → Alpine band (3000m)
- Northeast Entrance (670): 7420 ft (2262m) → Mid band (2500m)

Calibrated Parameters:
1. melt_factor: Degree-day factor (mm/°C/day) - controls melt rate
2. melt_thresh_temp: Temperature threshold for melt onset (°C)
3. precip_fraction: Controls rain/snow partitioning

Usage:
    python src/calibrate_snow_model.py
    python src/calibrate_snow_model.py --band alpine --station parker_peak
    python src/calibrate_snow_model.py --optimize
"""

import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional
from scipy.optimize import minimize, differential_evolution
from dataclasses import dataclass

# Add src to path if needed
sys.path.insert(0, str(Path(__file__).parent))


@dataclass
class SnowModelParams:
    """Snow model parameters for calibration"""
    melt_factor: float = 0.35          # mm/°C/day (degree-day factor)
    melt_thresh_temp: float = -6.0     # °C (temperature threshold for melt)
    precip_fraction: float = 0.167     # unitless (rain/snow partitioning)

    def to_array(self) -> np.ndarray:
        return np.array([self.melt_factor, self.melt_thresh_temp, self.precip_fraction])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'SnowModelParams':
        return cls(melt_factor=arr[0], melt_thresh_temp=arr[1], precip_fraction=arr[2])

    @staticmethod
    def get_bounds() -> list:
        """Parameter bounds for optimization"""
        return [
            (0.1, 1.5),      # melt_factor: typical range 1-6 mm/°C/day
            (-10.0, 2.0),    # melt_thresh_temp: -10 to +2°C
            (0.05, 0.3)      # precip_fraction: 0.05-0.3
        ]


class SnowModel:
    """Simplified snow model for calibration (matches water_balance.py SnowModel)"""

    def __init__(self, params: SnowModelParams):
        self.params = params

    def simulate(self, tmean: np.ndarray, precip: np.ndarray) -> np.ndarray:
        """
        Simulate SWE time series

        Args:
            tmean: Mean daily temperature (°C)
            precip: Daily precipitation (mm)

        Returns:
            swe: Simulated SWE time series (mm)
        """
        n_days = len(tmean)
        swe = np.zeros(n_days)

        for i in range(n_days):
            if i == 0:
                prev_swe = 0.0
            else:
                prev_swe = swe[i-1]

            # Melt
            if tmean[i] <= self.params.melt_thresh_temp:
                melt = 0.0
            else:
                melt = (tmean[i] - self.params.melt_thresh_temp) * self.params.melt_factor
                melt = min(melt, prev_swe)  # Can't melt more than available

            swe_after_melt = prev_swe - melt

            # Accumulation (rain/snow partitioning)
            if tmean[i] <= self.params.melt_thresh_temp:
                rain_fraction = 0.0
            elif tmean[i] > (self.params.melt_thresh_temp + 6):
                rain_fraction = 1.0
            else:
                rain_fraction = self.params.precip_fraction * (tmean[i] - self.params.melt_thresh_temp)

            rain_fraction = np.clip(rain_fraction, 0, 1)
            snow_increment = (1.0 - rain_fraction) * precip[i]

            swe[i] = swe_after_melt + snow_increment

        return swe


def load_snotel_data(snotel_file: str) -> pd.DataFrame:
    """Load SNOTEL SWE observations"""
    df = pd.read_csv(snotel_file)
    df['date'] = pd.to_datetime(df['date'])

    # Convert from inches to mm (1 inch = 25.4 mm)
    for col in df.columns:
        if col.startswith('swe_'):
            df[col] = df[col] * 25.4

    return df


def load_climate_data(climate_file: str) -> Dict[str, pd.DataFrame]:
    """Load climate data for elevation bands"""
    df = pd.read_csv(climate_file)
    df['date'] = pd.to_datetime(df['date'])

    bands = {}
    for band in ['valley', 'mid', 'alpine']:
        bands[band] = pd.DataFrame({
            'date': df['date'],
            'precip': df[f'precip_{band}'],
            'temp': df[f'temp_{band}']
        })

    return bands


def calculate_metrics(observed: np.ndarray, simulated: np.ndarray) -> Dict[str, float]:
    """Calculate performance metrics for SWE comparison"""
    # Remove NaN values
    mask = ~(np.isnan(observed) | np.isnan(simulated))
    obs = observed[mask]
    sim = simulated[mask]

    if len(obs) == 0:
        return {'nse': np.nan, 'rmse': np.nan, 'bias': np.nan, 'r2': np.nan}

    # NSE
    numerator = np.sum((obs - sim) ** 2)
    denominator = np.sum((obs - np.mean(obs)) ** 2)
    nse = 1 - (numerator / (denominator + 1e-10))

    # RMSE
    rmse = np.sqrt(np.mean((obs - sim) ** 2))

    # Bias (mm)
    bias = np.mean(sim - obs)

    # R²
    if np.std(obs) > 0 and np.std(sim) > 0:
        r2 = np.corrcoef(obs, sim)[0, 1] ** 2
    else:
        r2 = np.nan

    # Peak SWE metrics
    obs_peak = np.max(obs)
    sim_peak = np.max(sim)
    peak_error = sim_peak - obs_peak
    peak_error_pct = (peak_error / obs_peak) * 100 if obs_peak > 0 else np.nan

    # Peak timing (day of year for annual max - approximate)
    obs_peak_idx = np.argmax(obs)
    sim_peak_idx = np.argmax(sim)
    peak_timing_error = sim_peak_idx - obs_peak_idx  # days

    return {
        'nse': nse,
        'rmse': rmse,
        'bias': bias,
        'r2': r2,
        'peak_error_mm': peak_error,
        'peak_error_pct': peak_error_pct,
        'peak_timing_error_days': peak_timing_error
    }


def objective_function(params_array: np.ndarray,
                       tmean: np.ndarray,
                       precip: np.ndarray,
                       observed_swe: np.ndarray) -> float:
    """
    Objective function for optimization

    Minimizes: 1 - NSE (so optimal is 0)
    """
    params = SnowModelParams.from_array(params_array)
    model = SnowModel(params)
    simulated = model.simulate(tmean, precip)

    metrics = calculate_metrics(observed_swe, simulated)

    # Return 1 - NSE (we want to minimize, so lower is better)
    # Add penalty for unrealistic parameters
    penalty = 0.0
    if params.melt_factor < 0.1 or params.melt_factor > 2.0:
        penalty += 10.0

    return (1 - metrics['nse']) + penalty


def calibrate_parameters(tmean: np.ndarray,
                        precip: np.ndarray,
                        observed_swe: np.ndarray,
                        method: str = 'differential_evolution') -> Tuple[SnowModelParams, Dict]:
    """
    Calibrate snow model parameters using optimization

    Args:
        tmean: Mean daily temperature (°C)
        precip: Daily precipitation (mm)
        observed_swe: Observed SWE (mm)
        method: 'differential_evolution' or 'nelder-mead'

    Returns:
        Tuple of (optimal parameters, optimization result dict)
    """
    bounds = SnowModelParams.get_bounds()

    if method == 'differential_evolution':
        # Global optimization - more robust
        result = differential_evolution(
            objective_function,
            bounds=bounds,
            args=(tmean, precip, observed_swe),
            seed=42,
            maxiter=500,
            tol=1e-6,
            workers=1,  # Set to -1 for parallel
            updating='deferred',
            disp=True
        )
    else:
        # Local optimization from default starting point
        x0 = SnowModelParams().to_array()
        result = minimize(
            objective_function,
            x0,
            args=(tmean, precip, observed_swe),
            method='Nelder-Mead',
            options={'maxiter': 1000, 'disp': True}
        )

    optimal_params = SnowModelParams.from_array(result.x)

    return optimal_params, {
        'success': result.success,
        'nse': 1 - result.fun,
        'iterations': getattr(result, 'nit', None),
        'message': getattr(result, 'message', '')
    }


def plot_calibration_results(dates: np.ndarray,
                            observed: np.ndarray,
                            simulated_default: np.ndarray,
                            simulated_calibrated: np.ndarray,
                            default_params: SnowModelParams,
                            calibrated_params: SnowModelParams,
                            station_name: str,
                            band_name: str,
                            output_file: Optional[str] = None):
    """Create diagnostic plots comparing default vs calibrated simulations"""

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))

    # Calculate metrics
    metrics_default = calculate_metrics(observed, simulated_default)
    metrics_calibrated = calculate_metrics(observed, simulated_calibrated)

    # 1. Time series comparison (full)
    ax = axes[0, 0]
    ax.plot(dates, observed, 'k-', linewidth=1.5, label='Observed (SNOTEL)', alpha=0.8)
    ax.plot(dates, simulated_default, 'b--', linewidth=1, label=f'Default (NSE={metrics_default["nse"]:.3f})', alpha=0.7)
    ax.plot(dates, simulated_calibrated, 'r-', linewidth=1, label=f'Calibrated (NSE={metrics_calibrated["nse"]:.3f})', alpha=0.7)
    ax.set_ylabel('SWE (mm)')
    ax.set_title(f'{station_name} → {band_name.capitalize()} Band: Full Time Series')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # 2. Zoom on recent years (last 5 years)
    ax = axes[0, 1]
    recent_mask = dates >= (dates.max() - pd.Timedelta(days=5*365))
    ax.plot(dates[recent_mask], observed[recent_mask], 'k-', linewidth=1.5, label='Observed', alpha=0.8)
    ax.plot(dates[recent_mask], simulated_default[recent_mask], 'b--', linewidth=1, label='Default', alpha=0.7)
    ax.plot(dates[recent_mask], simulated_calibrated[recent_mask], 'r-', linewidth=1, label='Calibrated', alpha=0.7)
    ax.set_ylabel('SWE (mm)')
    ax.set_title('Recent 5 Years')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # 3. Scatter plot - Default
    ax = axes[1, 0]
    ax.scatter(observed, simulated_default, alpha=0.3, s=10, c='blue')
    max_val = max(observed.max(), simulated_default.max())
    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='1:1 line')
    ax.set_xlabel('Observed SWE (mm)')
    ax.set_ylabel('Simulated SWE (mm)')
    ax.set_title(f'Default Parameters\nNSE={metrics_default["nse"]:.3f}, RMSE={metrics_default["rmse"]:.1f}mm')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', 'box')

    # 4. Scatter plot - Calibrated
    ax = axes[1, 1]
    ax.scatter(observed, simulated_calibrated, alpha=0.3, s=10, c='red')
    max_val = max(observed.max(), simulated_calibrated.max())
    ax.plot([0, max_val], [0, max_val], 'k--', linewidth=2, label='1:1 line')
    ax.set_xlabel('Observed SWE (mm)')
    ax.set_ylabel('Simulated SWE (mm)')
    ax.set_title(f'Calibrated Parameters\nNSE={metrics_calibrated["nse"]:.3f}, RMSE={metrics_calibrated["rmse"]:.1f}mm')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', 'box')

    # 5. Annual peak comparison
    ax = axes[2, 0]
    years = pd.to_datetime(dates).year.unique()
    obs_peaks = []
    sim_def_peaks = []
    sim_cal_peaks = []

    for year in years:
        year_mask = pd.to_datetime(dates).year == year
        if year_mask.sum() > 0:
            obs_peaks.append(observed[year_mask].max())
            sim_def_peaks.append(simulated_default[year_mask].max())
            sim_cal_peaks.append(simulated_calibrated[year_mask].max())

    x = np.arange(len(years))
    width = 0.25
    ax.bar(x - width, obs_peaks, width, label='Observed', color='black', alpha=0.7)
    ax.bar(x, sim_def_peaks, width, label='Default', color='blue', alpha=0.7)
    ax.bar(x + width, sim_cal_peaks, width, label='Calibrated', color='red', alpha=0.7)
    ax.set_xlabel('Year')
    ax.set_ylabel('Peak SWE (mm)')
    ax.set_title('Annual Peak SWE Comparison')
    ax.set_xticks(x[::3])  # Show every 3rd year
    ax.set_xticklabels(years[::3], rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # 6. Parameter comparison table
    ax = axes[2, 1]
    ax.axis('off')

    table_data = [
        ['Parameter', 'Default', 'Calibrated', 'Units'],
        ['Melt Factor', f'{default_params.melt_factor:.3f}', f'{calibrated_params.melt_factor:.3f}', 'mm/°C/day'],
        ['Melt Threshold', f'{default_params.melt_thresh_temp:.1f}', f'{calibrated_params.melt_thresh_temp:.1f}', '°C'],
        ['Precip Fraction', f'{default_params.precip_fraction:.3f}', f'{calibrated_params.precip_fraction:.3f}', '-'],
        ['', '', '', ''],
        ['Metric', 'Default', 'Calibrated', ''],
        ['NSE', f'{metrics_default["nse"]:.4f}', f'{metrics_calibrated["nse"]:.4f}', ''],
        ['RMSE', f'{metrics_default["rmse"]:.1f}', f'{metrics_calibrated["rmse"]:.1f}', 'mm'],
        ['Bias', f'{metrics_default["bias"]:.1f}', f'{metrics_calibrated["bias"]:.1f}', 'mm'],
        ['Peak Error', f'{metrics_default["peak_error_pct"]:.1f}', f'{metrics_calibrated["peak_error_pct"]:.1f}', '%'],
    ]

    table = ax.table(cellText=table_data, loc='center', cellLoc='center',
                     colWidths=[0.35, 0.22, 0.22, 0.21])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # Color header rows
    for i in [0, 5]:
        for j in range(4):
            table[(i, j)].set_facecolor('#E6E6E6')
            table[(i, j)].set_text_props(weight='bold')

    ax.set_title('Parameter and Metric Comparison', fontsize=12, fontweight='bold', pad=20)

    plt.suptitle(f'Snow Model Calibration: {station_name} Station vs {band_name.capitalize()} Band',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved plot: {output_file}")

    plt.show()

    return fig


def run_calibration(snotel_file: str,
                    climate_file: str,
                    station_column: str,
                    band_name: str,
                    output_dir: str,
                    optimize: bool = True) -> Dict:
    """
    Run full calibration workflow

    Args:
        snotel_file: Path to SNOTEL data
        climate_file: Path to climate data
        station_column: Column name in SNOTEL file (e.g., 'swe_parker_peak')
        band_name: Elevation band to calibrate ('valley', 'mid', 'alpine')
        output_dir: Directory for output files
        optimize: Whether to run optimization (False = just compare)

    Returns:
        Dictionary with results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    station_name = station_column.replace('swe_', '').replace('_', ' ').title()

    print(f"\n{'='*70}")
    print(f"Calibrating {band_name.upper()} band using {station_name} SNOTEL")
    print(f"{'='*70}")

    # Load data
    print("\nLoading data...")
    snotel_df = load_snotel_data(snotel_file)
    climate_data = load_climate_data(climate_file)

    # Get band climate data
    band_df = climate_data[band_name]

    # Merge on date
    merged = pd.merge(snotel_df[['date', station_column]],
                      band_df, on='date', how='inner')
    merged = merged.dropna()

    print(f"  Date range: {merged['date'].min().date()} to {merged['date'].max().date()}")
    print(f"  Valid days: {len(merged):,}")

    # Extract arrays
    dates = merged['date'].values
    tmean = merged['temp'].values
    precip = merged['precip'].values
    observed_swe = merged[station_column].values

    # Default parameters simulation
    print("\nRunning default parameters...")
    default_params = SnowModelParams()
    default_model = SnowModel(default_params)
    simulated_default = default_model.simulate(tmean, precip)

    metrics_default = calculate_metrics(observed_swe, simulated_default)
    print(f"  Default NSE: {metrics_default['nse']:.4f}")
    print(f"  Default RMSE: {metrics_default['rmse']:.1f} mm")
    print(f"  Default Bias: {metrics_default['bias']:.1f} mm")

    # Optimization
    if optimize:
        print("\nOptimizing parameters (this may take a minute)...")
        calibrated_params, opt_result = calibrate_parameters(
            tmean, precip, observed_swe, method='differential_evolution'
        )

        print(f"\n  Optimization {'succeeded' if opt_result['success'] else 'failed'}!")
        print(f"  Final NSE: {opt_result['nse']:.4f}")
        print(f"\n  Calibrated Parameters:")
        print(f"    melt_factor:      {calibrated_params.melt_factor:.4f} mm/°C/day (was {default_params.melt_factor})")
        print(f"    melt_thresh_temp: {calibrated_params.melt_thresh_temp:.2f} °C (was {default_params.melt_thresh_temp})")
        print(f"    precip_fraction:  {calibrated_params.precip_fraction:.4f} (was {default_params.precip_fraction})")
    else:
        calibrated_params = default_params

    # Run calibrated simulation
    calibrated_model = SnowModel(calibrated_params)
    simulated_calibrated = calibrated_model.simulate(tmean, precip)

    metrics_calibrated = calculate_metrics(observed_swe, simulated_calibrated)

    # Calculate improvement
    nse_improvement = metrics_calibrated['nse'] - metrics_default['nse']
    rmse_improvement = metrics_default['rmse'] - metrics_calibrated['rmse']

    print(f"\n  Performance Improvement:")
    print(f"    NSE:  {metrics_default['nse']:.4f} → {metrics_calibrated['nse']:.4f} (Δ = {nse_improvement:+.4f})")
    print(f"    RMSE: {metrics_default['rmse']:.1f} → {metrics_calibrated['rmse']:.1f} mm (Δ = {rmse_improvement:+.1f} mm)")

    # Create diagnostic plots
    print("\nCreating diagnostic plots...")
    plot_file = output_path / f"calibration_{band_name}_{station_column.replace('swe_', '')}.png"

    plot_calibration_results(
        dates=pd.to_datetime(dates),
        observed=observed_swe,
        simulated_default=simulated_default,
        simulated_calibrated=simulated_calibrated,
        default_params=default_params,
        calibrated_params=calibrated_params,
        station_name=station_name,
        band_name=band_name,
        output_file=str(plot_file)
    )

    # Save calibrated parameters
    params_file = output_path / f"calibrated_params_{band_name}.json"
    import json
    params_dict = {
        'band': band_name,
        'station': station_column,
        'calibration_date': datetime.now().isoformat(),
        'parameters': {
            'melt_factor': calibrated_params.melt_factor,
            'melt_thresh_temp': calibrated_params.melt_thresh_temp,
            'precip_fraction': calibrated_params.precip_fraction
        },
        'default_parameters': {
            'melt_factor': default_params.melt_factor,
            'melt_thresh_temp': default_params.melt_thresh_temp,
            'precip_fraction': default_params.precip_fraction
        },
        'metrics': {
            'default': metrics_default,
            'calibrated': metrics_calibrated
        },
        'improvement': {
            'nse': nse_improvement,
            'rmse': rmse_improvement
        }
    }

    with open(params_file, 'w') as f:
        json.dump(params_dict, f, indent=2, default=float)
    print(f"  ✓ Saved parameters: {params_file}")

    return params_dict


def main():
    parser = argparse.ArgumentParser(
        description='Calibrate water balance snow model using SNOTEL data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--snotel-file',
                        default='data/snotel/lamar_snotel_combined.csv',
                        help='SNOTEL SWE data file')
    parser.add_argument('--climate-file',
                        default='data/climate/lamar_gridmet_elevation_bands.csv',
                        help='Climate data file with elevation bands')
    parser.add_argument('--output-dir',
                        default='output/calibration',
                        help='Output directory')
    parser.add_argument('--band',
                        choices=['alpine', 'mid', 'valley', 'all'],
                        default='all',
                        help='Elevation band to calibrate')
    parser.add_argument('--station',
                        choices=['parker_peak', 'northeast_entrance', 'all'],
                        default='all',
                        help='SNOTEL station to use')
    parser.add_argument('--no-optimize',
                        action='store_true',
                        help='Skip optimization (just compare default)')

    args = parser.parse_args()

    # Define station-to-band mappings
    station_band_mappings = {
        'parker_peak': 'alpine',      # 9420 ft → Alpine (3000m)
        'northeast_entrance': 'mid'   # 7420 ft → Mid (2500m)
    }

    results = {}

    if args.station == 'all' and args.band == 'all':
        # Calibrate both stations with their appropriate bands
        for station, band in station_band_mappings.items():
            result = run_calibration(
                snotel_file=args.snotel_file,
                climate_file=args.climate_file,
                station_column=f'swe_{station}',
                band_name=band,
                output_dir=args.output_dir,
                optimize=not args.no_optimize
            )
            results[f"{station}_{band}"] = result
    elif args.station != 'all':
        # Use specified station with appropriate or specified band
        band = args.band if args.band != 'all' else station_band_mappings[args.station]
        result = run_calibration(
            snotel_file=args.snotel_file,
            climate_file=args.climate_file,
            station_column=f'swe_{args.station}',
            band_name=band,
            output_dir=args.output_dir,
            optimize=not args.no_optimize
        )
        results[f"{args.station}_{band}"] = result
    else:
        # Specified band with all matching stations
        for station, mapped_band in station_band_mappings.items():
            if mapped_band == args.band:
                result = run_calibration(
                    snotel_file=args.snotel_file,
                    climate_file=args.climate_file,
                    station_column=f'swe_{station}',
                    band_name=args.band,
                    output_dir=args.output_dir,
                    optimize=not args.no_optimize
                )
                results[f"{station}_{args.band}"] = result

    # Summary
    print(f"\n{'='*70}")
    print("CALIBRATION SUMMARY")
    print(f"{'='*70}")

    for name, result in results.items():
        print(f"\n{name}:")
        print(f"  Default NSE:    {result['metrics']['default']['nse']:.4f}")
        print(f"  Calibrated NSE: {result['metrics']['calibrated']['nse']:.4f}")
        print(f"  Improvement:    {result['improvement']['nse']:+.4f}")
        print(f"  Parameters:")
        for param, value in result['parameters'].items():
            print(f"    {param}: {value:.4f}")

    print(f"\n✓ Calibration complete! Results saved to: {args.output_dir}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
