#!/usr/bin/env python3
"""
Download SNOTEL Snow Water Equivalent (SWE) data for Lamar River watershed.

SNOTEL Stations:
- 683: Parker Peak (elevation 8,510 ft / 2,594 m)
- 670: Northeast Entrance (elevation 7,520 ft / 2,292 m)

Data Source: USDA Natural Resources Conservation Service (NRCS)
API: NRCS AWDB (Air-Water Database) Web Service

Author: Stage 3 Enhancement
Date: 2025-11-26
"""

import os
import requests
import pandas as pd
from datetime import datetime, timedelta
import time
from pathlib import Path

# SNOTEL station configuration
# All stations within Lamar River watershed
STATIONS = {
    'northeast_entrance': {
        'id': '670',
        'name': 'Northeast Entrance',
        'state': 'MT',
        'elevation_ft': 7420,
        'elevation_m': 2262
    },
    'parker_peak': {
        'id': '683',
        'name': 'Parker Peak',
        'state': 'WY',
        'elevation_ft': 9420,
        'elevation_m': 2871
    }
}

# Date range (match streamflow data)
START_DATE = '1998-10-01'
END_DATE = '2025-11-26'

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / 'data' / 'snotel'


def get_snotel_data(station_id, start_date, end_date, state):
    """
    Download SNOTEL Snow Water Equivalent data from NRCS CSV interface.

    Args:
        station_id: SNOTEL station ID (e.g., '683')
        start_date: Start date string 'YYYY-MM-DD'
        end_date: End date string 'YYYY-MM-DD'
        state: Two letter state abbreviation

    Returns:
        DataFrame with columns: ['date', 'value']
    """
    # NRCS CSV download URL - use state from parameter
    url = (
        f"https://wcc.sc.egov.usda.gov/reportGenerator/view_csv/"
        f"customSingleStationReport/daily/"
        f"{station_id}:{state}:SNTL%7Cid=%22%22%7Cname/"
        f"{start_date},{end_date}/"
        f"WTEQ::value"
    )

    print(f"  Downloading from: {url}")

    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()

        # Parse CSV - skip comment lines that start with #
        lines = response.text.strip().split('\n')

        # Find the header row (contains "Date")
        header_idx = None
        for i, line in enumerate(lines):
            if not line.startswith('#') and 'Date' in line:
                header_idx = i
                break

        if header_idx is None:
            raise ValueError("Could not find header row in CSV")

        # Parse data starting from header
        data_text = '\n'.join(lines[header_idx:])
        df = pd.read_csv(pd.io.common.StringIO(data_text))

        # Clean column names
        df.columns = df.columns.str.strip()

        # Find date and value columns
        date_col = None
        value_col = None

        for col in df.columns:
            if 'Date' in col:
                date_col = col
            if 'Snow Water Equivalent' in col or 'WTEQ' in col:
                value_col = col

        if date_col is None or value_col is None:
            print(f"  Available columns: {df.columns.tolist()}")
            raise ValueError(f"Could not find required columns (date: {date_col}, value: {value_col})")

        # Rename and select columns
        df = df.rename(columns={date_col: 'date', value_col: 'value'})
        df = df[['date', 'value']].copy()

        # Convert types
        df['date'] = pd.to_datetime(df['date'])
        df['value'] = pd.to_numeric(df['value'], errors='coerce')

        # Sort and reset index
        df = df.sort_values('date').reset_index(drop=True)

        print(f"  ✓ Downloaded {len(df)} records")

        return df

    except Exception as e:
        print(f"  ✗ Download failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def download_station_data(station_key, station_info, start_date, end_date):
    """Download data for a single station."""

    print(f"\n{'='*80}")
    print(f"Station: {station_info['name']} ({station_info['id']})")
    print(f"State: {station_info['state']}")
    print(f"  Elevation: {station_info['elevation_ft']} ft ({station_info['elevation_m']} m)")
    print(f"  Date range: {start_date} to {end_date}")
    print(f"{'='*80}")

    # Download using CSV method
    df = get_snotel_data(station_info['id'], start_date, end_date, station_info['state'])

    if df is None or len(df) == 0:
        print(f"  ✗ Failed to download data for {station_info['name']}")
        return None

    # Add station info
    df['station_id'] = station_info['id']
    df['station_name'] = station_info['name']

    # Filter to date range
    df = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()

    # Summary statistics
    print(f"\n  Summary:")
    print(f"    Records: {len(df)}")
    print(f"    Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"    Missing values: {df['value'].isna().sum()} ({df['value'].isna().sum()/len(df)*100:.1f}%)")
    print(f"    SWE range: {df['value'].min():.1f} to {df['value'].max():.1f} inches")
    print(f"    Mean SWE: {df['value'].mean():.1f} inches")

    return df


def main():
    """Download SNOTEL data for all stations."""

    print("="*80)
    print("SNOTEL Data Downloader for Lamar River Watershed")
    print("="*80)
    print(f"\nDownloading Snow Water Equivalent (SWE) data...")
    print(f"Date range: {START_DATE} to {END_DATE}")
    print(f"Stations: {len(STATIONS)}")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {OUTPUT_DIR}")

    # Download each station
    station_data = {}

    for station_key, station_info in STATIONS.items():
        df = download_station_data(station_key, station_info, START_DATE, END_DATE)

        if df is not None:
            station_data[station_key] = df

            # Save individual station file
            output_file = OUTPUT_DIR / f"{station_key}_swe.csv"
            df[['date', 'value']].to_csv(output_file, index=False)
            print(f"  ✓ Saved to: {output_file}")

        # Be nice to the server
        time.sleep(1)

    # Combine all stations
    if len(station_data) > 0:
        print(f"\n{'='*80}")
        print("Combining all stations...")
        print(f"{'='*80}")

        # Create a combined dataframe
        # Wide format: date, swe_parker_peak, swe_northeast_entrance

        # Start with date range
        all_dates = pd.date_range(start=START_DATE, end=END_DATE, freq='D')
        df_combined = pd.DataFrame({'date': all_dates})

        # Add each station
        for station_key, df_station in station_data.items():
            # Rename value column
            df_station = df_station[['date', 'value']].copy()
            df_station = df_station.rename(columns={'value': f'swe_{station_key}'})

            # Merge
            df_combined = pd.merge(df_combined, df_station, on='date', how='left')

        # Fill missing values
        # Strategy: Forward fill for up to 7 days (common for SNOTEL data gaps)
        for col in df_combined.columns:
            if col != 'date':
                df_combined[col] = df_combined[col].ffill(limit=7)

        # Save combined file
        combined_file = OUTPUT_DIR / 'lamar_snotel_combined.csv'
        df_combined.to_csv(combined_file, index=False)

        print(f"\n✓ Combined file saved: {combined_file}")
        print(f"  Shape: {df_combined.shape}")
        print(f"  Columns: {df_combined.columns.tolist()}")
        print(f"\nSample data:")
        print(df_combined.head(10))

        # Summary statistics
        print(f"\n{'='*80}")
        print("Summary Statistics:")
        print(f"{'='*80}")
        print(df_combined.describe())

        # Data availability
        print(f"\n{'='*80}")
        print("Data Availability:")
        print(f"{'='*80}")
        for col in df_combined.columns:
            if col != 'date':
                n_valid = df_combined[col].notna().sum()
                pct_valid = n_valid / len(df_combined) * 100
                print(f"  {col}: {n_valid:,} / {len(df_combined):,} ({pct_valid:.1f}%)")

        print(f"\n{'='*80}")
        print("✓ SNOTEL download complete!")
        print(f"{'='*80}")

        return df_combined

    else:
        print(f"\n✗ Failed to download any station data")
        return None


if __name__ == '__main__':
    main()
