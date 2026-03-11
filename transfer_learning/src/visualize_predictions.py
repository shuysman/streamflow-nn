#!/usr/bin/env python3
"""
Visualize Model Predictions

Generate comprehensive plots for model evaluation:
- Full time series (train, validation, test)
- Test period hydrograph
- Scatter plot (observed vs predicted)
- Residual analysis
- Monthly performance

Usage:
    python src/visualize_predictions.py --run-dir runs/lamar_finetune_3var_0801_093233 --epoch 20
"""

import argparse
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import xarray as xr

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10


def load_results(run_dir, epoch):
    """Load prediction results from NeuralHydrology output"""
    results_file = run_dir / f'test/model_epoch{epoch:03d}/test_results.p'

    with open(results_file, 'rb') as f:
        results = pickle.load(f)

    return results


def extract_predictions(basin_results):
    """Extract dates, observations, and predictions from NeuralHydrology results"""
    # Results are stored in xarray Dataset under '1D' -> 'xr'
    xr_data = basin_results['1D']['xr']

    # Extract data
    dates = pd.to_datetime(xr_data['date'].values)
    obs = xr_data['QObs_mm_d_obs'].values.squeeze()  # Remove time_step dimension
    pred = xr_data['QObs_mm_d_sim'].values.squeeze()

    return dates, obs, pred


def calculate_metrics(obs, pred):
    """Calculate evaluation metrics"""
    # Remove NaNs
    mask = ~np.isnan(obs) & ~np.isnan(pred)
    obs = obs[mask]
    pred = pred[mask]

    # NSE
    nse = 1 - np.sum((obs - pred)**2) / np.sum((obs - np.mean(obs))**2)

    # RMSE
    rmse = np.sqrt(np.mean((obs - pred)**2))

    # MAE
    mae = np.mean(np.abs(obs - pred))

    # Bias
    bias = np.mean(pred - obs)

    # Pearson correlation
    corr = np.corrcoef(obs, pred)[0, 1]

    # R²
    r2 = corr**2

    return {
        'NSE': nse,
        'RMSE': rmse,
        'MAE': mae,
        'Bias': bias,
        'R²': r2,
        'Pearson-r': corr
    }


def plot_full_time_series(dates, obs, pred, config, metrics, output_file):
    """Plot full time series with train/val/test periods"""

    # Determine if we need the test detail panel
    val_end = pd.to_datetime(config['validation_end_date'])
    show_test_detail = dates.min() < val_end

    # Adjust layout based on content
    if show_test_detail:
        # 3-row layout: full series, test detail, scatter+residual
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
        ax1_pos = gs[0, :]
        ax3_pos = gs[2, 0]
        ax4_pos = gs[2, 1]
    else:
        # 2-row layout: full series, scatter+residual
        fig = plt.figure(figsize=(16, 7))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        ax1_pos = gs[0, :]
        ax3_pos = gs[1, 0]
        ax4_pos = gs[1, 1]

    # Main plot - full time series
    ax1 = fig.add_subplot(ax1_pos)

    ax1.plot(dates, obs, 'k-', linewidth=1, label='Observed', alpha=0.7)
    ax1.plot(dates, pred, 'r-', linewidth=1, label='Predicted', alpha=0.8)

    # Mark period boundaries
    train_end = pd.to_datetime(config['train_end_date'])
    train_start = pd.to_datetime(config['train_start_date'])

    # Only plot vertical lines if they fall within the visible date range
    if train_end >= dates.min() and train_end <= dates.max():
        ax1.axvline(train_end, color='blue', linestyle='--', alpha=0.5, linewidth=2, label='Train/Val split')
    if val_end >= dates.min() and val_end <= dates.max():
        ax1.axvline(val_end, color='green', linestyle='--', alpha=0.5, linewidth=2, label='Val/Test split')

    # Add period labels based on actual config dates (not data indices)
    # Calculate midpoints for each period
    train_mid = train_start + (train_end - train_start) / 2
    val_mid = train_end + (val_end - train_end) / 2
    test_mid = val_end + (dates[-1] - val_end) / 2

    # Only add labels if the midpoint falls within visible range
    y_pos = ax1.get_ylim()[1] * 0.95
    # if train_mid >= dates.min() and train_mid <= dates.max():
    #     ax1.text(train_mid, y_pos, 'Training', ha='center', fontsize=12, fontweight='bold',
    #             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    # if val_mid >= dates.min() and val_mid <= dates.max():
    #     ax1.text(val_mid, y_pos, 'Validation', ha='center', fontsize=12, fontweight='bold',
    #             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    # if test_mid >= dates.min() and test_mid <= dates.max():
    #     ax1.text(test_mid, y_pos, 'Test', ha='center', fontsize=12, fontweight='bold',
    #             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

    ax1.set_ylabel('Streamflow (mm/day)', fontsize=12, fontweight='bold')

    # Determine which period(s) are shown
    period_label = "Test Period Predictions"
    if dates.min() < val_end:
        if dates.min() < train_end:
            period_label = "Full Time Series"
        else:
            period_label = "Validation + Test Period"

    ax1.set_title(f"{config['experiment_name']} - {period_label}\nNSE: {metrics['NSE']:.3f} | RMSE: {metrics['RMSE']:.3f} mm/day",
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # Test period detail - only show if we have data beyond test period
    # (if dates only contains test data, this would be redundant)
    if show_test_detail:
        ax2 = fig.add_subplot(gs[1, :])
        test_mask = dates >= val_end
        test_dates = dates[test_mask]
        test_obs = obs[test_mask]
        test_pred = pred[test_mask]

        # Filter NaN values for all test period operations
        valid_mask = ~np.isnan(test_obs) & ~np.isnan(test_pred)
        test_dates_clean = test_dates[valid_mask]
        test_obs_clean = test_obs[valid_mask]
        test_pred_clean = test_pred[valid_mask]

        ax2.plot(test_dates_clean, test_obs_clean, 'k-', linewidth=1.5, label='Observed', alpha=0.7)
        ax2.plot(test_dates_clean, test_pred_clean, 'r-', linewidth=1.5, label='Predicted', alpha=0.8)
        ax2.fill_between(test_dates_clean, test_obs_clean, test_pred_clean, alpha=0.3, color='gray')

        test_metrics = calculate_metrics(test_obs_clean, test_pred_clean)
        ax2.set_ylabel('Streamflow (mm/day)', fontsize=12, fontweight='bold')
        ax2.set_title(f"Test Period (2020-2024) Detail\nNSE: {test_metrics['NSE']:.3f} | RMSE: {test_metrics['RMSE']:.3f} mm/day | Bias: {test_metrics['Bias']:.3f} mm/day",
                      fontsize=12, fontweight='bold')
        ax2.legend(loc='upper left', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    else:
        # Data is already test-only, use cleaned data for scatter/residual plots
        valid_mask = ~np.isnan(obs) & ~np.isnan(pred)
        test_dates_clean = dates[valid_mask]
        test_obs_clean = obs[valid_mask]
        test_pred_clean = pred[valid_mask]
        test_metrics = calculate_metrics(test_obs_clean, test_pred_clean)

    # Scatter plot
    ax3 = fig.add_subplot(ax3_pos)

    # Use cleaned test period data (NaNs removed)
    ax3.scatter(test_obs_clean, test_pred_clean, alpha=0.5, s=10, c='steelblue', edgecolors='none')

    # 1:1 line
    min_val = min(test_obs_clean.min(), test_pred_clean.min())
    max_val = max(test_obs_clean.max(), test_pred_clean.max())
    ax3.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='1:1 line')

    # Regression line
    z = np.polyfit(test_obs_clean, test_pred_clean, 1)
    p = np.poly1d(z)
    ax3.plot([min_val, max_val], [p(min_val), p(max_val)], 'r-', linewidth=2,
             label=f'Fit: y={z[0]:.2f}x+{z[1]:.2f}')

    ax3.set_xlabel('Observed (mm/day)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Predicted (mm/day)', fontsize=11, fontweight='bold')
    ax3.set_title(f"Scatter Plot (Test Period)\nR² = {test_metrics['R²']:.3f}", fontsize=11, fontweight='bold')
    ax3.legend(loc='upper left', fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal')

    # Residual plot
    ax4 = fig.add_subplot(ax4_pos)

    residuals = test_pred_clean - test_obs_clean
    ax4.scatter(test_obs_clean, residuals, alpha=0.5, s=10, c='coral', edgecolors='none')
    ax4.axhline(0, color='k', linestyle='--', linewidth=2)

    # Add mean and std bands
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)
    ax4.axhline(mean_residual, color='red', linestyle='-', linewidth=2, label=f'Mean: {mean_residual:.3f}')
    ax4.axhline(mean_residual + std_residual, color='red', linestyle=':', linewidth=1.5, alpha=0.7, label=f'±1σ: {std_residual:.3f}')
    ax4.axhline(mean_residual - std_residual, color='red', linestyle=':', linewidth=1.5, alpha=0.7)

    ax4.set_xlabel('Observed (mm/day)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Residual (mm/day)', fontsize=11, fontweight='bold')
    ax4.set_title(f"Residual Plot (Test Period)\nMAE = {test_metrics['MAE']:.3f} mm/day", fontsize=11, fontweight='bold')
    ax4.legend(loc='upper left', fontsize=9)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    plt.close()


def plot_monthly_performance(dates, obs, pred, config, output_file):
    """Plot monthly performance statistics"""

    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'obs': obs,
        'pred': pred
    })
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year

    # Filter to test period
    val_end = pd.to_datetime(config['validation_end_date'])
    df_test = df[df['date'] >= val_end].copy()

    # Calculate monthly metrics
    monthly_nse = []
    monthly_rmse = []
    months = []

    for month in range(1, 13):
        month_data = df_test[df_test['month'] == month]
        if len(month_data) > 10:  # Need enough data
            obs_month = month_data['obs'].values
            pred_month = month_data['pred'].values

            metrics = calculate_metrics(obs_month, pred_month)
            monthly_nse.append(metrics['NSE'])
            monthly_rmse.append(metrics['RMSE'])
            months.append(month)

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    # NSE by month
    colors = ['green' if nse > 0.7 else 'orange' if nse > 0.5 else 'red' for nse in monthly_nse]
    ax1.bar(months, monthly_nse, color=colors, alpha=0.7, edgecolor='black')
    ax1.axhline(0.7, color='green', linestyle='--', linewidth=2, label='Good (NSE > 0.7)')
    ax1.axhline(0.5, color='orange', linestyle='--', linewidth=2, label='Fair (NSE > 0.5)')
    ax1.set_ylabel('NSE', fontsize=12, fontweight='bold')
    ax1.set_title(f'Monthly Performance - {config["experiment_name"]} (Test Period)', fontsize=14, fontweight='bold')
    ax1.set_xticks(months)
    ax1.set_xticklabels([month_names[m-1] for m in months])
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim([min(monthly_nse) - 0.1, 1.0])

    # RMSE by month
    ax2.bar(months, monthly_rmse, color='steelblue', alpha=0.7, edgecolor='black')
    ax2.set_ylabel('RMSE (mm/day)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Month', fontsize=12, fontweight='bold')
    ax2.set_xticks(months)
    ax2.set_xticklabels([month_names[m-1] for m in months])
    ax2.grid(True, alpha=0.3, axis='y')

    # Add mean line
    mean_rmse = np.mean(monthly_rmse)
    ax2.axhline(mean_rmse, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_rmse:.3f}')
    ax2.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    plt.close()


def plot_flow_duration_curve(obs, pred, config, output_file):
    """Plot flow duration curve"""

    # Remove NaNs
    mask = ~np.isnan(obs) & ~np.isnan(pred)
    obs = obs[mask]
    pred = pred[mask]

    # Sort in descending order
    obs_sorted = np.sort(obs)[::-1]
    pred_sorted = np.sort(pred)[::-1]

    # Exceedance probability
    n = len(obs_sorted)
    exceedance = np.arange(1, n + 1) / n * 100

    fig, ax = plt.subplots(figsize=(12, 7))

    ax.semilogy(exceedance, obs_sorted, 'k-', linewidth=2, label='Observed')
    ax.semilogy(exceedance, pred_sorted, 'r-', linewidth=2, label='Predicted', alpha=0.8)

    # Add percentile markers
    percentiles = [5, 25, 50, 75, 95]
    for p in percentiles:
        idx = int(n * p / 100)
        if idx < n:
            ax.axvline(p, color='gray', linestyle=':', alpha=0.5)
            ax.text(p, obs_sorted[idx], f'Q{p}', fontsize=9, ha='center', va='bottom')

    ax.set_xlabel('Exceedance Probability (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Streamflow (mm/day)', fontsize=12, fontweight='bold')
    ax.set_title(f'Flow Duration Curve - {config["experiment_name"]}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize model predictions')
    parser.add_argument('--run-dir', required=True, help='Run directory path')
    parser.add_argument('--epoch', type=int, default=20, help='Epoch number to visualize')
    parser.add_argument('--output-dir', default='output/evaluation', help='Output directory for plots')
    parser.add_argument('--basin', type=str, default=None, help='Basin to visualize (default: first basin)')

    args = parser.parse_args()

    # Setup paths
    base_dir = Path(__file__).parent.parent
    run_dir = base_dir / args.run_dir
    output_dir = base_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Visualizing Predictions")
    print(f"{'='*60}")
    print(f"Run directory: {run_dir}")
    print(f"Epoch: {args.epoch}")

    # Load config
    import yaml
    config_file = run_dir / 'config.yml'
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    print(f"Experiment: {config['experiment_name']}")

    # Load results
    print(f"\nLoading results...")
    results = load_results(run_dir, args.epoch)

    # Extract data for the specified basin
    if args.basin:
        basin = args.basin
        if basin not in results:
            available = list(results.keys())
            raise ValueError(f"Basin '{basin}' not found. Available basins: {available}")
    else:
        basin = list(results.keys())[0]

    print(f"Basin: {basin}")
    print(f"Available basins: {list(results.keys())}")

    basin_results = results[basin]
    dates, obs, pred = extract_predictions(basin_results)

    print(f"Date range: {dates.min()} to {dates.max()}")
    print(f"Records: {len(dates)}")

    # Calculate overall metrics
    metrics = calculate_metrics(obs, pred)
    print(f"\n{'='*60}")
    print(f"Overall Metrics:")
    print(f"{'='*60}")
    for metric, value in metrics.items():
        print(f"  {metric:12s}: {value:.4f}")

    # Generate plots
    print(f"\n{'='*60}")
    print(f"Generating Plots")
    print(f"{'='*60}")

    exp_name = config['experiment_name']

    # 1. Full time series
    output_file = output_dir / f'{exp_name}_full_timeseries.png'
    print(f"\n1. Creating full time series plot...")
    plot_full_time_series(dates, obs, pred, config, metrics, output_file)

    # 2. Monthly performance
    output_file = output_dir / f'{exp_name}_monthly_performance.png'
    print(f"\n2. Creating monthly performance plot...")
    plot_monthly_performance(dates, obs, pred, config, output_file)

    # 3. Flow duration curve
    output_file = output_dir / f'{exp_name}_flow_duration.png'
    print(f"\n3. Creating flow duration curve...")
    plot_flow_duration_curve(obs, pred, config, output_file)

    print(f"\n{'='*60}")
    print(f"✅ All plots created successfully!")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")


if __name__ == '__main__':
    main()
