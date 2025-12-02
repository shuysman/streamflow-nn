#!/usr/bin/env python3
"""
Download USGS sub-daily (15-minute/hourly) stream gauge data for Hoh River

The dataretrieval.nwis package has get_iv() for instantaneous values:
- get_dv() = Daily Values (what we've been using)
- get_iv() = Instantaneous Values (typically 15-minute intervals)

This higher temporal resolution may help capture rapid rainfall events
(Noah Effect) that are difficult to predict in rain-dominated systems.
"""

import dataretrieval.nwis as nwis
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# USGS Site for Hoh River
SITE_CODE = "12041200"
PARAMETER_CODE = "00060"  # Discharge (streamflow)

print("=" * 80)
print("USGS INSTANTANEOUS VALUES (SUB-DAILY) DOWNLOADER - HOH RIVER")
print("=" * 80)
print(f"\nSite: Hoh River near Forks, Washington")
print(f"USGS Site Code: {SITE_CODE}")
print(f"Parameter: Streamflow (CFS)")
print(f"\nData Type: Instantaneous Values (typically 15-minute intervals)")

# Start with a small test - get last 30 days of sub-daily data
end_date = datetime.now()
start_date = end_date - timedelta(days=30)

print(f"\nTest Download: Last 30 days")
print(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
print("\nDownloading instantaneous values...")
print("(This may take a moment...)")

try:
    # get_iv() retrieves instantaneous values
    df, metadata = nwis.get_iv(
        sites=SITE_CODE,
        parameterCd=PARAMETER_CODE,
        start=start_date.strftime('%Y-%m-%d'),
        end=end_date.strftime('%Y-%m-%d')
    )

    print(f"\n✓ Successfully downloaded {len(df)} records")
    print(f"✓ Date range: {df.index.min()} to {df.index.max()}")

    # Examine the data structure
    print("\n" + "=" * 80)
    print("DATA STRUCTURE")
    print("=" * 80)
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nFirst few records:")
    print(df.head(10))

    # Calculate time intervals
    time_diffs = df.index.to_series().diff()
    print(f"\n" + "=" * 80)
    print("TEMPORAL RESOLUTION")
    print("=" * 80)
    print(f"\nTime interval statistics:")
    print(time_diffs.describe())
    print(f"\nMost common interval: {time_diffs.mode()[0] if len(time_diffs.mode()) > 0 else 'N/A'}")

    # Process the data
    flow_col = [col for col in df.columns if '00060' in col][0]

    df_clean = pd.DataFrame({
        'datetime': df.index,
        'streamflow_cfs': df[flow_col].values
    }).reset_index(drop=True)

    # Remove missing/negative values
    df_clean = df_clean[df_clean['streamflow_cfs'] > 0]

    # Convert to m³/s
    df_clean['streamflow'] = df_clean['streamflow_cfs'] * 0.0283168

    print(f"\n✓ Clean dataset: {len(df_clean)} records")

    # Statistics
    print("\n" + "=" * 80)
    print("STREAMFLOW STATISTICS (Last 30 days)")
    print("=" * 80)
    print(f"\nStreamflow (m³/s):")
    print(df_clean['streamflow'].describe())

    # Calculate records per day
    records_per_day = len(df_clean) / 30
    interval_minutes = 1440 / records_per_day  # 1440 minutes in a day

    print(f"\n✓ Average records per day: {records_per_day:.1f}")
    print(f"✓ Average interval: {interval_minutes:.1f} minutes")

    # Save sample data
    output_file = '../data/hoh_river_hourly_sample.csv'
    df_clean.to_csv(output_file, index=False)
    print(f"\n✓ Sample data saved to: {output_file}")

    # Visualization
    print("\nGenerating visualization...")
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Full 30 days
    axes[0].plot(df_clean['datetime'], df_clean['streamflow'], linewidth=0.5, alpha=0.7)
    axes[0].set_title('Hoh River Sub-Daily Streamflow (Last 30 Days)', fontweight='bold')
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Streamflow (m³/s)')
    axes[0].grid(True, alpha=0.3)

    # Last 7 days - zoomed in
    recent = df_clean[df_clean['datetime'] >= df_clean['datetime'].max() - pd.Timedelta(days=7)]
    axes[1].plot(recent['datetime'], recent['streamflow'], linewidth=1, marker='o',
                 markersize=2, alpha=0.8)
    axes[1].set_title('Last 7 Days (Zoomed) - Shows Sub-Daily Resolution', fontweight='bold')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Streamflow (m³/s)')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../data/hoh_river_hourly_visualization.png', dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved")
    plt.close()

    # Information about getting more data
    print("\n" + "=" * 80)
    print("RETRIEVING HISTORICAL SUB-DAILY DATA")
    print("=" * 80)
    print("\nTo get longer periods of sub-daily data:")
    print("  1. Use get_iv() with desired date range")
    print("  2. For very long periods, download in chunks (e.g., year by year)")
    print("  3. Be aware: sub-daily data files can be VERY large")
    print("     - 15-min intervals = 96 records/day = 35,040 records/year")
    print("     - 65 years (like our daily data) = 2.28 million records!")

    print("\n" + "=" * 80)
    print("WHY SUB-DAILY DATA FOR HOH RIVER?")
    print("=" * 80)
    print("\nRain-dominated systems (Noah Effect):")
    print("  • Rapid storm events cause sudden flow changes")
    print("  • Daily averages smooth out these peaks")
    print("  • Sub-daily data captures the actual hydrograph shape")
    print("  • May improve prediction of rainfall-driven peaks")
    print("\nSnowmelt systems (Joseph Effect):")
    print("  • Daily data is usually sufficient")
    print("  • Gradual, predictable melt patterns")
    print("  • Less benefit from higher temporal resolution")

    print("\n" + "=" * 80)
    print(f"SUCCESS! {len(df_clean)} instantaneous records retrieved")
    print("=" * 80)

except Exception as e:
    print(f"\n✗ Error: {e}")
    print("\nPossible reasons:")
    print("  1. Not all USGS sites have instantaneous data")
    print("  2. Instantaneous data may not cover full historical period")
    print("  3. USGS servers may be temporarily unavailable")
    print("\nTo check data availability for a site:")
    print(f"  https://waterdata.usgs.gov/nwis/uv?site_no={SITE_CODE}")
