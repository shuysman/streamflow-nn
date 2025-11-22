#!/usr/bin/env python3
"""
Download USGS stream gauge data for Hoh River, Olympic National Park, Washington

The Hoh River is a temperate rainforest river - very different from the
snowmelt-dominated Lamar River. This will test model generalization.
"""

import dataretrieval.nwis as nwis
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# USGS Site for Hoh River near Forks, WA
SITE_CODE = "12041200"  # Hoh River near Forks, WA
PARAMETER_CODE = "00060"  # Discharge (streamflow) in cubic feet per second

print("=" * 80)
print("USGS Stream Gauge Data Downloader - HOH RIVER")
print("=" * 80)
print(f"\nSite: Hoh River near Forks, Washington")
print(f"USGS Site Code: {SITE_CODE}")
print(f"Parameter: Streamflow (CFS)")
print(f"\nClimate: Temperate rainforest (rain-dominated)")
print(f"vs Lamar River: Mountain snowmelt-dominated")

# Get site information
print("\nFetching site information...")
try:
    site_info = nwis.get_info(sites=SITE_CODE)[0]
    print(f"✓ Site Name: {site_info['station_nm']}")
    print(f"✓ Location: {site_info['dec_lat_va']}°N, {site_info['dec_long_va']}°W")
    print(f"✓ Drainage Area: {site_info.get('drain_area_va', 'N/A')} sq mi")
except Exception as e:
    print(f"⚠ Could not fetch site info: {e}")

# Download daily streamflow data
print(f"\nDownloading daily streamflow data...")
print("This may take a minute...")

try:
    # Get all available data
    df, metadata = nwis.get_dv(
        sites=SITE_CODE,
        parameterCd=PARAMETER_CODE,
        start='1900-01-01',  # Get all available data
        end=datetime.now().strftime('%Y-%m-%d')
    )

    print(f"✓ Successfully downloaded {len(df)} days of data")
    print(f"✓ Date range: {df.index.min()} to {df.index.max()}")

    # Process the data
    print("\nProcessing data...")

    # The data comes with a column like '00060_Mean' for mean daily streamflow
    flow_col = [col for col in df.columns if '00060' in col][0]

    # Create clean dataframe
    df_clean = pd.DataFrame({
        'date': df.index,
        'streamflow_cfs': df[flow_col].values
    }).reset_index(drop=True)

    # Remove USGS missing data codes (negative values)
    df_clean = df_clean[df_clean['streamflow_cfs'] > 0]

    # Convert CFS to cubic meters per second (m³/s)
    df_clean['streamflow'] = df_clean['streamflow_cfs'] * 0.0283168

    # Remove any NaN values
    df_clean = df_clean.dropna()

    print(f"✓ Clean dataset: {len(df_clean)} days")
    print(f"✓ Date range: {df_clean['date'].min()} to {df_clean['date'].max()}")

    # Statistics
    print("\n" + "=" * 80)
    print("STREAMFLOW STATISTICS")
    print("=" * 80)
    print(f"\nIn Cubic Feet per Second (CFS):")
    print(df_clean['streamflow_cfs'].describe())
    print(f"\nIn Cubic Meters per Second (m³/s):")
    print(df_clean['streamflow'].describe())

    # Check for missing data
    total_days = (df_clean['date'].max() - df_clean['date'].min()).days
    missing_pct = (1 - len(df_clean) / total_days) * 100
    print(f"\nData completeness: {100-missing_pct:.1f}% ({len(df_clean)}/{total_days} days)")

    # Save data
    output_file = '../data/hoh_river_streamflow.csv'
    df_clean.to_csv(output_file, index=False)
    print(f"\n✓ Data saved to: {output_file}")

    # Create visualization
    print("\nGenerating visualization...")
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # Full time series
    axes[0].plot(df_clean['date'], df_clean['streamflow'], linewidth=0.5, alpha=0.7)
    axes[0].set_title('Hoh River Daily Streamflow - Full Record', fontweight='bold')
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Streamflow (m³/s)')
    axes[0].grid(True, alpha=0.3)

    # Recent 5 years
    recent_df = df_clean[df_clean['date'] >= df_clean['date'].max() - pd.Timedelta(days=365*5)]
    axes[1].plot(recent_df['date'], recent_df['streamflow'], linewidth=1, alpha=0.8)
    axes[1].set_title('Recent 5 Years', fontweight='bold')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Streamflow (m³/s)')
    axes[1].grid(True, alpha=0.3)

    # Monthly statistics (boxplot by month)
    df_clean['month'] = pd.to_datetime(df_clean['date']).dt.month
    monthly_data = [df_clean[df_clean['month'] == m]['streamflow'].values for m in range(1, 13)]
    bp = axes[2].boxplot(monthly_data, labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                          patch_artist=True)
    axes[2].set_title('Monthly Streamflow Distribution', fontweight='bold')
    axes[2].set_xlabel('Month')
    axes[2].set_ylabel('Streamflow (m³/s)')
    axes[2].grid(True, alpha=0.3, axis='y')

    # Color the boxplots
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)

    plt.tight_layout()
    plt.savefig('../data/hoh_river_visualization.png', dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved to: hoh_river_visualization.png")
    plt.close()

    # Compare to Lamar River
    print("\n" + "=" * 80)
    print("COMPARISON: HOH vs LAMAR RIVER")
    print("=" * 80)
    print("\nHoh River (Rainforest, Washington):")
    print(f"  Mean flow: {df_clean['streamflow'].mean():.2f} m³/s")
    print(f"  Peak month: {['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][df_clean.groupby('month')['streamflow'].mean().idxmax()-1]}")
    print(f"  Climate: Rain-dominated, year-round precipitation")

    print("\nLamar River (Mountain, Wyoming/Montana):")
    print(f"  Mean flow: ~1.79 m³/s")
    print(f"  Peak month: May-Jun (snowmelt)")
    print(f"  Climate: Snowmelt-dominated, semi-arid")

    print("\n" + "=" * 80)
    print("DATA READY FOR MODEL TRAINING!")
    print("=" * 80)
    print(f"\nFile: {output_file}")
    print(f"Records: {len(df_clean)} days ({len(df_clean)/365:.1f} years)")
    print(f"Columns: date, streamflow_cfs, streamflow (m³/s)")

except Exception as e:
    print(f"\n✗ Error downloading data: {e}")
    print("\nTroubleshooting:")
    print("  1. Check internet connection")
    print("  2. Verify USGS site code is correct")
    print("  3. USGS servers may be temporarily unavailable")
