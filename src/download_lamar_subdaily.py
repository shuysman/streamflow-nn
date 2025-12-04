#!/usr/bin/env python3
"""
Download sub-daily (15-minute) Lamar River data for comparison with Hoh River.

Question: Will subdaily resolution improve Lamar's already-high R² = 0.97?
- Snowmelt is a slow process (may not benefit from higher resolution)
- But diurnal melt patterns might be informative
"""

import dataretrieval.nwis as nwis
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys

SITE_CODE = "06188000"  # Lamar River nr Tower Ranger Station YNP 
#SITE_CODE = "06187915"  # Soda Butte Cr at Park Bndry at Silver Gate
PARAMETER_CODE = "00060"

print("=" * 80)
print("DOWNLOADING LAMAR RIVER SUB-DAILY DATA")
print("=" * 80)

# First, check if subdaily data exists
print("\nChecking for subdaily data availability...")

test_start = datetime(2020, 1, 1)
test_end = test_start + timedelta(days=7)

try:
    df_test, _ = nwis.get_iv(
        sites=SITE_CODE,
        parameterCd=PARAMETER_CODE,
        start=test_start.strftime('%Y-%m-%d'),
        end=test_end.strftime('%Y-%m-%d')
    )

    if len(df_test) > 0:
        print(f"✓ Subdaily data IS available!")
        print(f"  Found {len(df_test)} records in test week")
        interval = df_test.index.to_series().diff().median()
        print(f"  Typical interval: {interval}")
    else:
        print("✗ No subdaily data found for this site")
        print("\nLamar River may only have daily data.")
        print("This is common for remote wilderness sites.")
        sys.exit(1)

except Exception as e:
    print(f"✗ Error checking for subdaily data: {e}")
    print("\nSubdaily data likely not available for this site.")
    sys.exit(1)

# Download from 1990 onwards (or as far back as available)
print("\n" + "=" * 80)
print("DOWNLOADING FROM 1990 (OR AS FAR BACK AS AVAILABLE)")
print("=" * 80)

end_date = datetime.now()
start_date = datetime(1990, 1, 1)
total_years = end_date.year - start_date.year + 1

print(f"\nDate range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
print("Downloading in yearly chunks...\n")

all_data = []
successful_years = 0
failed_years = 0

for year in range(start_date.year, end_date.year + 1):
    chunk_start = datetime(year, 1, 1)
    chunk_end = datetime(year, 12, 31) if year < end_date.year else end_date

    print(f"Year {year} ({year - start_date.year + 1}/{total_years})...", end=" ", flush=True)

    try:
        df, metadata = nwis.get_iv(
            sites=SITE_CODE,
            parameterCd=PARAMETER_CODE,
            start=chunk_start.strftime('%Y-%m-%d'),
            end=chunk_end.strftime('%Y-%m-%d')
        )

        if len(df) > 0:
            print(f"✓ {len(df):,} records")
            all_data.append(df)
            successful_years += 1
        else:
            print("⚠ No data")
            failed_years += 1

    except Exception as e:
        print(f"✗ Error: {e}")
        failed_years += 1

# Combine all chunks
print("\n" + "=" * 80)
print("COMBINING AND PROCESSING DATA")
print("=" * 80)

if len(all_data) == 0:
    print("\n✗ No data downloaded!")
    print("\nLamar River may only have daily data, not subdaily.")
    sys.exit(1)

print(f"\n✓ Successfully downloaded {successful_years} years")
if failed_years > 0:
    print(f"⚠ Failed/no data for {failed_years} years")

print("\nCombining chunks...")
df_all = pd.concat(all_data).sort_index()

# Remove duplicates
df_all = df_all[~df_all.index.duplicated(keep='first')]

print(f"✓ Total records: {len(df_all):,}")
print(f"✓ Date range: {df_all.index.min()} to {df_all.index.max()}")

# Process the data
flow_col = [col for col in df_all.columns if '00060' in col][0]

df_clean = pd.DataFrame({
    'datetime': df_all.index,
    'streamflow_cfs': df_all[flow_col].values
}).reset_index(drop=True)

# Remove missing/negative values
initial_count = len(df_clean)
df_clean = df_clean[df_clean['streamflow_cfs'] > 0]
df_clean = df_clean.dropna()
removed_count = initial_count - len(df_clean)

print(f"✓ Removed {removed_count:,} missing/invalid records ({removed_count/initial_count*100:.2f}%)")

# Convert to m³/s
df_clean['streamflow'] = df_clean['streamflow_cfs'] * 0.0283168

# Calculate actual interval
df_clean['datetime'] = pd.to_datetime(df_clean['datetime'])
intervals = df_clean['datetime'].diff()
median_interval = intervals.median()

print(f"\n✓ Clean dataset: {len(df_clean):,} records")
print(f"✓ Median interval: {median_interval}")
print(f"✓ Records per day: {pd.Timedelta('1 day') / median_interval:.1f}")

# Statistics
print("\n" + "=" * 80)
print("STREAMFLOW STATISTICS")
print("=" * 80)
print(f"\nStreamflow (m³/s):")
print(df_clean['streamflow'].describe())

# Check for gaps
print("\n" + "=" * 80)
print("DATA QUALITY CHECK")
print("=" * 80)

# Find gaps > 1 hour
gaps = intervals[intervals > pd.Timedelta('1 hour')]
if len(gaps) > 0:
    print(f"\n⚠ Found {len(gaps):,} gaps > 1 hour")
    print(f"  Largest gap: {gaps.max()}")
    print(f"  Gaps 1-6 hours: {len(gaps[gaps <= pd.Timedelta('6 hours')]):,}")
    print(f"  Gaps 6-24 hours: {len(gaps[(gaps > pd.Timedelta('6 hours')) & (gaps <= pd.Timedelta('1 day'))]):,}")
    print(f"  Gaps > 24 hours: {len(gaps[gaps > pd.Timedelta('1 day')]):,}")
else:
    print("\n✓ No significant gaps in data")

# Save
output_file = '../data/lamar_river_subdaily.csv'
print(f"\nSaving to {output_file}...")
df_clean.to_csv(output_file, index=False)
print(f"✓ Data saved")

import os
file_size_mb = os.path.getsize(output_file) / 1024 / 1024
print(f"✓ File size: {file_size_mb:.1f} MB")

print("\n" + "=" * 80)
print("DATA READY FOR DEEP LEARNING MODEL")
print("=" * 80)
print(f"\nRecords: {len(df_clean):,}")
print(f"Time span: {(df_clean['datetime'].max() - df_clean['datetime'].min()).days} days ({(df_clean['datetime'].max() - df_clean['datetime'].min()).days/365.25:.1f} years)")
print(f"Temporal resolution: ~{median_interval.total_seconds()/60:.0f} minutes")

print("\n" + "=" * 80)
print("COMPARISON TO DAILY MODEL")
print("=" * 80)
print("\nDaily data model (lamar_river_model.ipynb):")
print("  R² ≈ 0.97 (already excellent!)")
print("  Lag-1 autocorrelation = 0.986 (extremely high)")
print("\nQuestion: Can subdaily data improve on this?")
print("  • Snowmelt is slow (diurnal cycle might be weak)")
print("  • Daily data already captures the smooth melt pattern")
print("  • Subdaily may show diurnal temperature-driven variation")
print("  • Or it may just add noise without improving predictions")
print("\n→ Train subdaily model to find out!")

print("\n" + "=" * 80)
