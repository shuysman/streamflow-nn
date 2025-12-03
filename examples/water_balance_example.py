#!/usr/bin/env python3
"""
Example script showing how to use the water_balance module

This demonstrates both:
1. Loading data from a CSV file
2. Running the water balance model
3. Saving results and creating plots
"""

import sys
import datetime
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from water_balance import (
    WaterBalanceConfig,
    run_water_balance,
    save_results_to_csv,
    plot_water_balance
)


def load_csv_data(filepath: str) -> dict:
    """
    Load weather data from CSV file

    Expected CSV format:
        date,tmax,tmin,precip
        01/01/2020,45.5,32.1,0.15
        ...

    Or with headers that include these columns.
    """
    df = pd.read_csv(filepath)

    # Normalize column names
    df.columns = df.columns.str.lower().str.strip()

    # Convert to dict format expected by water_balance
    data = {
        'date': df['date'].values,
        'tmax': df['tmax'].values,
        'tmin': df['tmin'].values,
        'precip': df['precip'].values
    }

    return data


def example_basic_usage():
    """Example 1: Basic water balance calculation"""

    # Create configuration
    config = WaterBalanceConfig(
        station='ExampleStation',
        year_start=2020,
        year_end=2022,
        station_type='GHCN',
        pet_type='hamon',
        max_soil_water=100.0,
        latitude=45.5,  # North latitude
        elevation=1000.0,  # meters
        use_heatload=True,
        forgive_snow=True,
        title='Example Station'
    )

    # Load data (replace with your actual data file)
    # data = load_csv_data('path/to/your/data.csv')

    # For demonstration, create synthetic data
    start_date = datetime.datetime(2020, 1, 1)
    end_date = datetime.datetime(2022, 12, 31)
    n_days = (end_date - start_date).days + 1

    dates = []
    current = start_date
    one_day = datetime.timedelta(days=1)
    while current <= end_date:
        dates.append(f"{current.month:02d}/{current.day:02d}/{current.year}")
        current += one_day

    # Synthetic weather data
    data = {
        'date': np.array(dates),
        'tmax': np.random.normal(60, 15, n_days),  # Fahrenheit
        'tmin': np.random.normal(40, 15, n_days),
        'precip': np.abs(np.random.normal(0.1, 0.15, n_days))  # inches
    }

    print("Running water balance model...")

    # Run model
    results = run_water_balance(data, config, start_date, end_date)

    print(f"✓ Processed {len(results.dates)} days")
    print(f"  Mean PET: {np.nanmean(results.pet):.2f} mm/day")
    print(f"  Mean AET: {np.nanmean(results.aet):.2f} mm/day")
    print(f"  Mean soil water: {np.nanmean(results.soil_water):.2f} mm")
    print(f"  Total runoff: {np.nansum(results.runoff):.2f} mm")

    # Save results
    output_dir = Path(__file__).parent / 'output'
    output_dir.mkdir(exist_ok=True)

    csv_path = output_dir / 'water_balance_results.csv'
    save_results_to_csv(results, str(csv_path))
    print(f"\n✓ Saved results to {csv_path}")

    # Create plot
    plot_path = output_dir / 'water_balance_plot.png'
    plot_water_balance(results, config.title, str(plot_path))
    print(f"✓ Saved plot to {plot_path}")

    return results


def example_compare_pet_methods():
    """Example 2: Compare different PET calculation methods"""

    start_date = datetime.datetime(2021, 1, 1)
    end_date = datetime.datetime(2021, 12, 31)
    n_days = (end_date - start_date).days + 1

    # Create synthetic data
    dates = []
    current = start_date
    one_day = datetime.timedelta(days=1)
    while current <= end_date:
        dates.append(f"{current.month:02d}/{current.day:02d}/{current.year}")
        current += one_day

    data = {
        'date': np.array(dates),
        'tmax': np.random.normal(65, 20, n_days),
        'tmin': np.random.normal(45, 20, n_days),
        'precip': np.abs(np.random.normal(0.1, 0.15, n_days))
    }

    pet_methods = ['hamon', 'hargreaves', 'oudin']
    results_dict = {}

    print("\nComparing PET methods for 2021...")

    for method in pet_methods:
        config = WaterBalanceConfig(
            station='ComparisonStation',
            year_start=2021,
            year_end=2021,
            pet_type=method,
            latitude=45.0,
            elevation=500.0,
            title=f'{method.capitalize()} Method'
        )

        results = run_water_balance(data, config, start_date, end_date)
        results_dict[method] = results

        print(f"\n{method.upper()}:")
        print(f"  Mean annual PET: {np.nansum(results.pet):.1f} mm")
        print(f"  Mean annual AET: {np.nansum(results.aet):.1f} mm")
        print(f"  Mean annual deficit: {np.nansum(results.deficit):.1f} mm")

    return results_dict


def example_accessing_results():
    """Example 3: Accessing and analyzing results"""

    # Quick setup
    config = WaterBalanceConfig(
        station='Test',
        year_start=2021,
        year_end=2021,
        latitude=45.0,
        elevation=500.0
    )

    start_date = datetime.datetime(2021, 1, 1)
    end_date = datetime.datetime(2021, 12, 31)
    n_days = (end_date - start_date).days + 1

    dates = []
    current = start_date
    one_day = datetime.timedelta(days=1)
    while current <= end_date:
        dates.append(f"{current.month:02d}/{current.day:02d}/{current.year}")
        current += one_day

    data = {
        'date': np.array(dates),
        'tmax': np.random.normal(65, 20, n_days),
        'tmin': np.random.normal(45, 20, n_days),
        'precip': np.abs(np.random.normal(0.1, 0.15, n_days))
    }

    results = run_water_balance(data, config, start_date, end_date)

    print("\nAccessing results as arrays:")
    print(f"  Type: {type(results)}")
    print(f"  Dates: {results.dates[:5]} ... (first 5)")
    print(f"  PET shape: {results.pet.shape}")

    print("\nConverting to dictionary:")
    results_dict = results.to_dict()
    print(f"  Keys: {list(results_dict.keys())}")

    print("\nConverting to pandas DataFrame:")
    import pandas as pd
    df = pd.DataFrame(results.to_dict())
    print(df.head())

    print("\nCalculating monthly statistics:")
    df['month'] = pd.to_datetime(df['dates'], format='%m/%d/%Y').dt.month
    monthly_stats = df.groupby('month').agg({
        'pet': 'sum',
        'aet': 'sum',
        'd': 'sum',
        'runoff': 'sum'
    })
    print(monthly_stats)

    return df


def main():
    """Run all examples"""

    print("="*70)
    print("Water Balance Module - Usage Examples")
    print("="*70)

    print("\n" + "="*70)
    print("Example 1: Basic Usage")
    print("="*70)
    results1 = example_basic_usage()

    print("\n" + "="*70)
    print("Example 2: Compare PET Methods")
    print("="*70)
    results2 = example_compare_pet_methods()

    print("\n" + "="*70)
    print("Example 3: Accessing Results")
    print("="*70)
    df = example_accessing_results()

    print("\n" + "="*70)
    print("All examples completed successfully!")
    print("="*70)


if __name__ == '__main__':
    main()
