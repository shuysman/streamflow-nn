#!/usr/bin/env python3
"""
Download SNOTEL data for Hoh River watershed sites.

Sites (from find_hoh_snotel_sites.py):
- 1107: Buckinghorse (4860 ft / 1481 m) - 35.7 km from watershed
- 974: Waterhole (5010 ft / 1527 m) - 38.6 km from watershed
- 1126: Mt. Tebo (3370 ft / 1027 m) - 54.2 km from watershed
"""

import pandas as pd
import requests
from pathlib import Path
from datetime import datetime

# Hoh SNOTEL stations
STATIONS = {
    'buckinghorse': {
        'id': '1107',
        'name': 'Buckinghorse',
        'state': 'WA',
        'elevation_ft': 4860
    },
    'waterhole': {
        'id': '974',
        'name': 'Waterhole',
        'state': 'WA',
        'elevation_ft': 5010
    },
    'mt_tebo': {
        'id': '1126',
        'name': 'Mt. Tebo',
        'state': 'WA',
        'elevation_ft': 3370
    }
}

START_DATE = '2010-01-01'
END_DATE = '2025-01-01'
OUTPUT_DIR = Path('../data/snotel')

def download_snotel_data(station_id, state, start_date, end_date):
    """
    Download SNOTEL SWE data from NRCS.

    Uses the reportGenerator CSV endpoint.
    """
    url = (
        f"https://wcc.sc.egov.usda.gov/reportGenerator/view_csv/"
        f"customSingleStationReport/daily/{station_id}:{state}:SNTL%7Cid=%22%22%7Cname/"
        f"{start_date},{end_date}/WTEQ::value"
    )

    print(f"  Downloading from NRCS...")
    print(f"  URL: {url[:100]}...")

    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()

        # Parse CSV (NRCS format has header lines starting with #)
        lines = response.text.split('\\n')
        data_lines = [line for line in lines if not line.startswith('#') and line.strip()]

        if len(data_lines) < 2:
            print(f"  ⚠️  No data returned")
            return None

        # Convert to DataFrame
        from io import StringIO
        df = pd.read_csv(StringIO('\\n'.join(data_lines)))

        # Rename columns
        df.columns = ['date', 'swe_inches']
        df['date'] = pd.to_datetime(df['date'])

        # Convert inches to mm
        df['swe_mm'] = df['swe_inches'] * 25.4

        # Remove missing values
        df = df[df['swe_inches'] != -99.9].copy()

        print(f"  ✓ Downloaded {len(df)} days")
        print(f"    Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"    SWE range: {df['swe_mm'].min():.1f} - {df['swe_mm'].max():.1f} mm")

        return df

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return None

def main():
    """Download data for all Hoh SNOTEL sites."""
    print("="*80)
    print("HOH RIVER SNOTEL DOWNLOADER")
    print("="*80)
    print(f"Date range: {START_DATE} to {END_DATE}")
    print(f"Stations: {len(STATIONS)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_data = {}

    for key, station in STATIONS.items():
        print(f"\\n{'='*80}")
        print(f"Station: {station['name']} (ID: {station['id']})")
        print(f"Elevation: {station['elevation_ft']} ft")
        print("="*80)

        df = download_snotel_data(
            station_id=station['id'],
            state=station['state'],
            start_date=START_DATE,
            end_date=END_DATE
        )

        if df is not None:
            # Save individual station
            output_file = OUTPUT_DIR / f"{key}_swe.csv"
            df.to_csv(output_file, index=False)
            print(f"  ✓ Saved to: {output_file}")

            # Store for combined file
            all_data[f'swe_{key}'] = df.set_index('date')['swe_mm']

    # Create combined file
    if all_data:
        print(f"\\n{'='*80}")
        print("COMBINING DATA")
        print("="*80)

        combined = pd.DataFrame(all_data)
        combined.index.name = 'date'

        # Fill forward missing values (snow persists)
        combined = combined.fillna(method='ffill').fillna(0)

        # Add average
        combined['swe_avg'] = combined.mean(axis=1)

        output_file = OUTPUT_DIR / 'hoh_snotel_combined.csv'
        combined.to_csv(output_file)

        print(f"✓ Combined data: {len(combined)} days")
        print(f"  Columns: {list(combined.columns)}")
        print(f"✓ Saved to: {output_file}")

        # Summary statistics
        print(f"\\n{'='*80}")
        print("SUMMARY STATISTICS")
        print("="*80)
        print(combined.describe())

        print(f"\\n{'='*80}")
        print("✅ SUCCESS! SNOTEL data ready for Stage 6")
        print("="*80)
        print("\\nNext step:")
        print("  Update Stage 6 notebook to include climate + SNOTEL features")
    else:
        print(f"\\n✗ No data downloaded!")

if __name__ == '__main__':
    main()
