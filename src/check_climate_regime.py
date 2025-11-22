#!/usr/bin/env python3
"""
Check if 2015-2025 was unusually wet or dry compared to full record.
This tests if we're only capturing one Joseph Effect regime.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load full subdaily data
df = pd.read_csv('../data/hoh_river_subdaily_full.csv')
df['datetime'] = pd.to_datetime(df['datetime'])

# Aggregate to annual mean flow
df['year'] = df['datetime'].dt.year
annual_flow = df.groupby('year')['streamflow'].mean()

print("=" * 80)
print("CLIMATE REGIME ANALYSIS - Hoh River (1990-2025)")
print("=" * 80)

# Full record statistics
full_mean = annual_flow.mean()
full_std = annual_flow.std()

print(f"\nFull Record (1990-2025):")
print(f"  Mean annual flow: {full_mean:.2f} m³/s")
print(f"  Std deviation: {full_std:.2f} m³/s")

# Last 10 years (2015-2025)
recent_flow = annual_flow[annual_flow.index >= 2015]
recent_mean = recent_flow.mean()
recent_std = recent_flow.std()

print(f"\nLast 10 Years (2015-2025):")
print(f"  Mean annual flow: {recent_mean:.2f} m³/s")
print(f"  Std deviation: {recent_std:.2f} m³/s")

# Calculate z-score
z_score = (recent_mean - full_mean) / full_std

print(f"\n" + "=" * 80)
print("REGIME BIAS ASSESSMENT")
print("=" * 80)

print(f"\nZ-score of recent period: {z_score:.2f}")

if abs(z_score) < 0.5:
    print("✓ Recent period is REPRESENTATIVE of full record")
    print("  → 10-year dataset should generalize reasonably well")
elif z_score > 0.5:
    bias_pct = ((recent_mean - full_mean) / full_mean) * 100
    print(f"⚠️  Recent period is WETTER than average (+{bias_pct:.1f}%)")
    print("  → Model trained on wet regime may fail in dry periods")
    print("  → Joseph Effect: Model expects high-flow persistence")
elif z_score < -0.5:
    bias_pct = ((full_mean - recent_mean) / full_mean) * 100
    print(f"⚠️  Recent period is DRIER than average (-{bias_pct:.1f}%)")
    print("  → Model trained on dry regime may fail in wet periods")
    print("  → Joseph Effect: Model expects low-flow persistence")

# Check for trends (linear regression)
from scipy import stats
years = annual_flow.index.values
flows = annual_flow.values
slope, intercept, r_value, p_value, std_err = stats.linregress(years, flows)

print(f"\n" + "=" * 80)
print("LONG-TERM TREND ANALYSIS")
print("=" * 80)

trend_per_decade = slope * 10
print(f"\nTrend: {trend_per_decade:+.2f} m³/s per decade")
print(f"R² = {r_value**2:.3f}, p-value = {p_value:.4f}")

if p_value < 0.05:
    if slope > 0:
        print("✓ Significant INCREASING trend (wetter over time)")
    else:
        print("✓ Significant DECREASING trend (drier over time)")
    print("  → Recent 10 years may not represent historical conditions")
else:
    print("✓ No significant trend")
    print("  → Climate relatively stationary (good for modeling)")

# Identify wet vs dry periods
threshold_wet = full_mean + 0.5 * full_std
threshold_dry = full_mean - 0.5 * full_std

wet_years = annual_flow[annual_flow > threshold_wet]
dry_years = annual_flow[annual_flow < threshold_dry]

print(f"\n" + "=" * 80)
print("WET/DRY PERIOD DISTRIBUTION")
print("=" * 80)

print(f"\nWet years (>{threshold_wet:.1f} m³/s): {len(wet_years)}")
print(f"  Years: {list(wet_years.index)}")

print(f"\nDry years (<{threshold_dry:.1f} m³/s): {len(dry_years)}")
print(f"  Years: {list(dry_years.index)}")

# Check recent period composition
recent_wet = len([y for y in wet_years.index if y >= 2015])
recent_dry = len([y for y in dry_years.index if y >= 2015])
recent_normal = len(recent_flow) - recent_wet - recent_dry

print(f"\nLast 10 years composition:")
print(f"  Wet years: {recent_wet}/10 ({recent_wet*10}%)")
print(f"  Normal years: {recent_normal}/10 ({recent_normal*10}%)")
print(f"  Dry years: {recent_dry}/10 ({recent_dry*10}%)")

if recent_wet > 7:
    print("\n⚠️  HEAVILY WET-BIASED - Model will expect persistence of high flows")
elif recent_dry > 7:
    print("\n⚠️  HEAVILY DRY-BIASED - Model will expect persistence of low flows")
elif recent_wet > 5 or recent_dry > 5:
    print("\n⚠️  MODERATELY BIASED - Some regime bias present")
else:
    print("\n✓ BALANCED - Good mix of wet/dry conditions")

# Visualization
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Annual flow time series
axes[0].plot(annual_flow.index, annual_flow.values, 'o-', linewidth=2, markersize=6, label='Annual Mean Flow')
axes[0].axhline(y=full_mean, color='blue', linestyle='--', linewidth=2, label=f'Full Mean ({full_mean:.1f} m³/s)')
axes[0].axhline(y=threshold_wet, color='red', linestyle=':', linewidth=1.5, alpha=0.7, label='Wet Threshold')
axes[0].axhline(y=threshold_dry, color='orange', linestyle=':', linewidth=1.5, alpha=0.7, label='Dry Threshold')

# Highlight 2015-2025 period
axes[0].axvspan(2015, 2025, alpha=0.2, color='yellow', label='Training Period (2015-2025)')

# Add trend line
trend_line = slope * years + intercept
axes[0].plot(years, trend_line, 'r--', linewidth=2, alpha=0.5, label=f'Trend ({trend_per_decade:+.2f}/decade)')

axes[0].set_xlabel('Year', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Mean Annual Flow (m³/s)', fontsize=12, fontweight='bold')
axes[0].set_title('Hoh River Annual Mean Flow - Joseph Effect Assessment', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# Distribution comparison
recent_years = [y for y in annual_flow.index if y >= 2015]
historical_years = [y for y in annual_flow.index if y < 2015]

historical_flows = annual_flow[historical_years].values
recent_flows_vals = annual_flow[recent_years].values

axes[1].hist([historical_flows, recent_flows_vals], bins=15, label=['Historical (1990-2014)', 'Recent (2015-2025)'],
             alpha=0.7, edgecolor='black')
axes[1].axvline(x=full_mean, color='blue', linestyle='--', linewidth=2, label='Full Mean')
axes[1].axvline(x=recent_mean, color='red', linestyle='--', linewidth=2, label='Recent Mean')
axes[1].set_xlabel('Annual Mean Flow (m³/s)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Frequency (years)', fontsize=12, fontweight='bold')
axes[1].set_title('Distribution Comparison: Are We Training on a Biased Period?', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('../data/climate_regime_analysis.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Visualization saved to: ../data/climate_regime_analysis.png")
plt.show()

print("\n" + "=" * 80)
print("RECOMMENDATION")
print("=" * 80)

if abs(z_score) > 0.5:
    print("\n⚠️  STRONG RECOMMENDATION: Use full 36-year dataset")
    print("   Reasons:")
    print("   1. Recent period is biased relative to historical conditions")
    print("   2. Joseph Effect requires capturing full wet/dry cycles")
    print("   3. Model trained on one regime won't generalize to opposite regime")
    print("   4. 36 years better captures climate variability")
else:
    print("\n✓ 10-year dataset appears representative")
    print("  However, for robustness:")
    print("  • Consider using full dataset for final model")
    print("  • Test on both wet and dry holdout periods")
    print("  • Monitor performance across climate regimes")

print("\n" + "=" * 80)
