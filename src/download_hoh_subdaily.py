#!/usr/bin/env python3
"""
Download ALL available sub-daily (15-minute) Hoh River data

Note: Sub-daily data typically starts later than daily data (often 2000s-2010s)
We'll download in yearly chunks to avoid timeouts and memory issues.
"""

import dataretrieval.nwis as nwis
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys

SITE_CODE = "12041200"
PARAMETER_CODE = "00060"

print("=" * 80)
print("DOWNLOADING ALL HOH RIVER SUB-DAILY DATA")
print("=" * 80)

# First, try to determine when sub-daily data starts
# We'll try starting from 1990 and work our way forward
print("\nFinding start date of sub-daily data...")

test_years = [1990, 1995, 2000, 2005, 2010, 2015, 2020]
data_start = None

for year in test_years:
    try:
        test_start = datetime(year, 1, 1)
        test_end = test_start + timedelta(days=7)  # Just test 1 week

        df_test, _ = nwis.get_iv(
            sites=SITE_CODE,
            parameterCd=PARAMETER_CODE,
            start=test_start.strftime('%Y-%m-%d'),
            end=test_end.strftime('%Y-%m-%d')
        )

        if len(df_test) > 0:
            print(f"  ✓ Data available from {year}")
            data_start = year
            break
        else:
            print(f"  ✗ No data in {year}")
    except:
        print(f"  ✗ No data in {year}")

if data_start is None:
    print("\n✗ Could not find sub-daily data. It may not be available for this site.")
    sys.exit(1)

# Now download all data from data_start to now in yearly chunks
print(f"\n✓ Starting download from {data_start}")

end_date = datetime.now()
start_date = datetime(data_start, 1, 1)
total_years = end_date.year - start_date.year + 1

print(f"\nDownloading {total_years} years of data in yearly chunks...")
print("This will take several minutes...\n")

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
    sys.exit(1)

print(f"\n✓ Successfully downloaded {successful_years} years")
if failed_years > 0:
    print(f"⚠ Failed/no data for {failed_years} years")

print("\nCombining chunks...")
df_all = pd.concat(all_data).sort_index()

# Remove duplicates (in case of overlap at year boundaries)
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
    # Show distribution of gap sizes
    print(f"  Gaps 1-6 hours: {len(gaps[gaps <= pd.Timedelta('6 hours')]):,}")
    print(f"  Gaps 6-24 hours: {len(gaps[(gaps > pd.Timedelta('6 hours')) & (gaps <= pd.Timedelta('1 day'))]):,}")
    print(f"  Gaps > 24 hours: {len(gaps[gaps > pd.Timedelta('1 day')]):,}")
else:
    print("\n✓ No significant gaps in data")

# Save
output_file = '../data/hoh_river_subdaily_full.csv'
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
print(f"\nDaily data model (R² ≈ 0.66) used ~23,792 records")
print(f"Sub-daily model will use {len(df_clean):,} records ({len(df_clean)/23792:.1f}x more data points)")
print("\n" + "=" * 80)
