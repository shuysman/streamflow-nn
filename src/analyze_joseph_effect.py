#!/usr/bin/env python3
"""
Analyze the Joseph Effect (long-term persistence) in streamflow data.

This script:
1. Calculates autocorrelation at multiple timescales
2. Estimates the Hurst exponent (H) using R/S analysis
3. Shows what the model CAN vs CANNOT learn based on SEQ_LENGTH
4. Compares timescales: model lookback vs persistence timescale
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("JOSEPH EFFECT ANALYSIS - Long-Range Persistence in Streamflow")
print("=" * 80)

# ============================================================================
# Load Data
# ============================================================================

print("\nLoading data...")

# Hoh River (daily)
df_hoh = pd.read_csv('../data/hoh_river_streamflow.csv')
df_hoh['date'] = pd.to_datetime(df_hoh['date'])
hoh_flow = df_hoh['streamflow'].values

print(f"✓ Hoh River: {len(hoh_flow)} days ({len(hoh_flow)/365.25:.1f} years)")

# Try to load Lamar River for comparison
try:
    df_lamar = pd.read_csv('../data/lamar_river_streamflow.csv')
    df_lamar['date'] = pd.to_datetime(df_lamar['date'])
    lamar_flow = df_lamar['streamflow'].values
    print(f"✓ Lamar River: {len(lamar_flow)} days ({len(lamar_flow)/365.25:.1f} years)")
    has_lamar = True
except:
    print("⚠ Lamar River data not found (will analyze Hoh only)")
    has_lamar = False

# ============================================================================
# Function: Autocorrelation Function (ACF)
# ============================================================================

def calculate_acf(data, max_lag=365*5):
    """Calculate autocorrelation function up to max_lag."""
    n = len(data)
    mean = np.mean(data)
    var = np.var(data)

    acf = np.zeros(max_lag + 1)
    acf[0] = 1.0  # Perfect correlation at lag 0

    for lag in range(1, min(max_lag + 1, n)):
        c0 = np.sum((data[:n-lag] - mean) * (data[lag:] - mean)) / n
        acf[lag] = c0 / var

    return acf

# ============================================================================
# Function: Hurst Exponent via R/S Analysis (Rescaled Range)
# ============================================================================

def hurst_rs(data, min_lag=8, max_lag=None):
    """
    Calculate Hurst exponent using Rescaled Range (R/S) analysis.

    The Hurst exponent H characterizes long-term memory:
    - H = 0.5: Random walk (no persistence)
    - H > 0.5: Persistent (Joseph Effect - trends continue)
    - H < 0.5: Anti-persistent (mean-reverting)

    Returns:
        H: Hurst exponent
        lags: Array of lags used
        rs: Rescaled range values
    """
    if max_lag is None:
        max_lag = len(data) // 4

    lags = []
    rs_values = []

    # Try different window sizes
    for lag in range(min_lag, max_lag, max(1, max_lag // 50)):
        if lag >= len(data):
            break

        # Divide data into chunks of size 'lag'
        n_chunks = len(data) // lag
        if n_chunks < 2:
            continue

        rs_chunk = []
        for i in range(n_chunks):
            chunk = data[i*lag:(i+1)*lag]

            # Mean-adjusted cumulative sum
            mean_chunk = np.mean(chunk)
            Y = np.cumsum(chunk - mean_chunk)

            # Range
            R = np.max(Y) - np.min(Y)

            # Standard deviation
            S = np.std(chunk, ddof=1)

            if S > 0:
                rs_chunk.append(R / S)

        if len(rs_chunk) > 0:
            lags.append(lag)
            rs_values.append(np.mean(rs_chunk))

    lags = np.array(lags)
    rs_values = np.array(rs_values)

    # Fit line: log(R/S) = H * log(lag) + c
    # Hurst exponent is the slope
    log_lags = np.log(lags)
    log_rs = np.log(rs_values)

    H, intercept = np.polyfit(log_lags, log_rs, 1)

    return H, lags, rs_values

# ============================================================================
# Function: Decorrelation Time (e-folding time)
# ============================================================================

def decorrelation_time(acf, threshold=1/np.e):
    """
    Find the lag where autocorrelation drops below threshold (default: 1/e ≈ 0.37).
    This is the "memory" or "persistence" timescale.
    """
    for lag, r in enumerate(acf):
        if r < threshold:
            return lag
    return len(acf)  # Never decorrelates

# ============================================================================
# Analyze Hoh River
# ============================================================================

print("\n" + "=" * 80)
print("HOH RIVER ANALYSIS (Rain-Dominated)")
print("=" * 80)

# Autocorrelation
max_lag_days = min(365 * 5, len(hoh_flow) - 1)  # 5 years or max available
acf_hoh = calculate_acf(hoh_flow, max_lag=max_lag_days)

# Hurst exponent
H_hoh, lags_hoh, rs_hoh = hurst_rs(hoh_flow)

# Decorrelation time
decorr_hoh = decorrelation_time(acf_hoh)

print(f"\n1. Hurst Exponent (H): {H_hoh:.3f}")
if H_hoh > 0.7:
    print(f"   → STRONG Joseph Effect (high persistence)")
    print(f"   → Trends persist over long periods")
elif H_hoh > 0.55:
    print(f"   → MODERATE Joseph Effect (some persistence)")
    print(f"   → Noticeable long-term memory")
elif H_hoh > 0.45:
    print(f"   → WEAK persistence (near random walk)")
    print(f"   → Limited long-term memory")
else:
    print(f"   → Anti-persistent (mean-reverting)")

print(f"\n2. Key Autocorrelation Values:")
print(f"   Lag 1 day:    r = {acf_hoh[1]:.3f}  ← What persistence model learns")
print(f"   Lag 7 days:   r = {acf_hoh[7]:.3f}  ← Hourly model sees this")
print(f"   Lag 30 days:  r = {acf_hoh[30]:.3f}")
print(f"   Lag 60 days:  r = {acf_hoh[60]:.3f}  ← Daily model sees this")
print(f"   Lag 365 days: r = {acf_hoh[365]:.3f}  ← Annual persistence (Joseph Effect)")

print(f"\n3. Decorrelation Time: {decorr_hoh} days ({decorr_hoh/365.25:.2f} years)")
print(f"   → This is how long the 'memory' lasts")

print(f"\n4. Model Limitations:")
print(f"   Daily model SEQ_LENGTH = 60 days")
print(f"   → Can learn autocorrelation up to lag {60}")
print(f"   → Captures r = {acf_hoh[60]:.3f} of the signal")
print(f"   → MISSES {decorr_hoh - 60} days of persistence ({(decorr_hoh - 60)/decorr_hoh*100:.1f}% of memory)")

if has_lamar:
    print("\n" + "=" * 80)
    print("LAMAR RIVER ANALYSIS (Snowmelt-Dominated)")
    print("=" * 80)

    # Autocorrelation
    max_lag_days_lamar = min(365 * 5, len(lamar_flow) - 1)
    acf_lamar = calculate_acf(lamar_flow, max_lag=max_lag_days_lamar)

    # Hurst exponent
    H_lamar, lags_lamar, rs_lamar = hurst_rs(lamar_flow)

    # Decorrelation time
    decorr_lamar = decorrelation_time(acf_lamar)

    print(f"\n1. Hurst Exponent (H): {H_lamar:.3f}")
    if H_lamar > 0.7:
        print(f"   → STRONG Joseph Effect (high persistence)")
    elif H_lamar > 0.55:
        print(f"   → MODERATE Joseph Effect")
    else:
        print(f"   → WEAK persistence")

    print(f"\n2. Key Autocorrelation Values:")
    print(f"   Lag 1 day:    r = {acf_lamar[1]:.3f}")
    print(f"   Lag 7 days:   r = {acf_lamar[7]:.3f}")
    print(f"   Lag 30 days:  r = {acf_lamar[30]:.3f}")
    print(f"   Lag 60 days:  r = {acf_lamar[60]:.3f}")
    print(f"   Lag 365 days: r = {acf_lamar[365]:.3f}")

    print(f"\n3. Decorrelation Time: {decorr_lamar} days ({decorr_lamar/365.25:.2f} years)")

# ============================================================================
# Comparison
# ============================================================================

if has_lamar:
    print("\n" + "=" * 80)
    print("COMPARISON: Why Lamar Gets Higher R² Than Hoh")
    print("=" * 80)

    print(f"\nLag-1 Autocorrelation (day-to-day persistence):")
    print(f"  Lamar: r = {acf_lamar[1]:.3f}  → Very high (smooth snowmelt)")
    print(f"  Hoh:   r = {acf_hoh[1]:.3f}  → Lower (sudden storms)")
    print(f"  Difference: {(acf_lamar[1] - acf_hoh[1]):.3f}")

    print(f"\nModel Performance Explanation:")
    if acf_lamar[1] > acf_hoh[1]:
        print(f"  ✓ Lamar's higher lag-1 correlation ({acf_lamar[1]:.3f}) explains R² ≈ 0.97")
        print(f"  ✓ 'Tomorrow ≈ today' works very well for snowmelt")
        print(f"  ✓ Hoh's lower lag-1 correlation ({acf_hoh[1]:.3f}) explains R² ≈ 0.66")
        print(f"  ✓ Storms break the 'tomorrow ≈ today' pattern (Noah Effect)")

    print(f"\nJoseph Effect (multi-year persistence):")
    print(f"  Lamar H = {H_lamar:.3f}")
    print(f"  Hoh   H = {H_hoh:.3f}")
    print(f"  → Both have long-term persistence, but models DON'T learn this")
    print(f"  → Models only learn what fits in SEQ_LENGTH window")

# ============================================================================
# Visualizations
# ============================================================================

print("\n" + "=" * 80)
print("GENERATING VISUALIZATIONS")
print("=" * 80)

if has_lamar:
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
else:
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# ============================================================================
# Plot 1: Autocorrelation Function (HOH)
# ============================================================================

ax1 = fig.add_subplot(gs[0, 0])

lags_plot = np.arange(len(acf_hoh))
ax1.plot(lags_plot, acf_hoh, linewidth=2, color='blue', alpha=0.8)
ax1.axhline(y=1/np.e, color='red', linestyle='--', linewidth=2, label=f'1/e threshold (decorr = {decorr_hoh} days)')
ax1.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)

# Mark model lookback windows
ax1.axvspan(0, 7, alpha=0.2, color='yellow', label='Hourly model (7 days)')
ax1.axvspan(0, 60, alpha=0.15, color='orange', label='Daily model (60 days)')

ax1.set_xlabel('Lag (days)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Autocorrelation', fontsize=12, fontweight='bold')
ax1.set_title('Hoh River: Autocorrelation Function', fontsize=14, fontweight='bold')
ax1.set_xlim(0, min(365*3, len(acf_hoh)))
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=9)

# Add annotations
ax1.annotate(f'Lag 1: {acf_hoh[1]:.3f}', xy=(1, acf_hoh[1]), xytext=(50, acf_hoh[1]+0.1),
            arrowprops=dict(arrowstyle='->', color='black'), fontsize=10)
ax1.annotate(f'Lag 365: {acf_hoh[365]:.3f}\n(Joseph Effect)', xy=(365, acf_hoh[365]),
            xytext=(500, acf_hoh[365]+0.15),
            arrowprops=dict(arrowstyle='->', color='red'), fontsize=10, color='red')

# ============================================================================
# Plot 2: Hurst Exponent R/S Plot (HOH)
# ============================================================================

ax2 = fig.add_subplot(gs[0, 1])

log_lags_hoh = np.log(lags_hoh)
log_rs_hoh = np.log(rs_hoh)

ax2.scatter(log_lags_hoh, log_rs_hoh, alpha=0.6, s=30, label='Data')

# Fit line
fit_hoh = np.polyfit(log_lags_hoh, log_rs_hoh, 1)
fit_line_hoh = fit_hoh[0] * log_lags_hoh + fit_hoh[1]
ax2.plot(log_lags_hoh, fit_line_hoh, 'r-', linewidth=2, label=f'H = {H_hoh:.3f}')

# Reference lines
ax2.plot(log_lags_hoh, 0.5 * log_lags_hoh + fit_hoh[1], 'g--', linewidth=1.5, alpha=0.5, label='H = 0.5 (random)')
ax2.plot(log_lags_hoh, 0.7 * log_lags_hoh + fit_hoh[1], 'b--', linewidth=1.5, alpha=0.5, label='H = 0.7 (strong Joseph)')

ax2.set_xlabel('log(lag)', fontsize=12, fontweight='bold')
ax2.set_ylabel('log(R/S)', fontsize=12, fontweight='bold')
ax2.set_title(f'Hoh River: Hurst Exponent Analysis\nH = {H_hoh:.3f}', fontsize=14, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# ============================================================================
# Plot 3: Short-term vs Long-term ACF (HOH)
# ============================================================================

ax3 = fig.add_subplot(gs[1, 0])

# Short-term (what model sees)
short_lags = np.arange(0, min(180, len(acf_hoh)))
ax3.bar(short_lags, acf_hoh[:len(short_lags)], color='blue', alpha=0.7, edgecolor='black', linewidth=0.5)
ax3.axvline(x=7, color='yellow', linestyle='-', linewidth=3, alpha=0.7, label='Hourly model limit (7 days)')
ax3.axvline(x=60, color='orange', linestyle='-', linewidth=3, alpha=0.7, label='Daily model limit (60 days)')

ax3.set_xlabel('Lag (days)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Autocorrelation', fontsize=12, fontweight='bold')
ax3.set_title('What the Model CAN Learn (Hoh)', fontsize=14, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3, axis='y')

# ============================================================================
# Plot 4: What Model Misses (HOH)
# ============================================================================

ax4 = fig.add_subplot(gs[1, 1])

# Plot beyond model's lookback
long_lags = np.arange(60, min(365*5, len(acf_hoh)))
ax4.plot(long_lags, acf_hoh[long_lags], linewidth=2, color='red', alpha=0.8)
ax4.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
ax4.axhline(y=1/np.e, color='green', linestyle='--', linewidth=2, alpha=0.5, label=f'Decorrelation at {decorr_hoh} days')

# Shade the "missed" region
ax4.fill_between(long_lags, 0, acf_hoh[long_lags], alpha=0.3, color='red', label='Joseph Effect (missed)')

ax4.set_xlabel('Lag (days)', fontsize=12, fontweight='bold')
ax4.set_ylabel('Autocorrelation', fontsize=12, fontweight='bold')
ax4.set_title('What the Model CANNOT Learn (Hoh)', fontsize=14, fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)

# Mark annual cycle
if len(acf_hoh) > 365:
    ax4.annotate(f'1 year: {acf_hoh[365]:.3f}', xy=(365, acf_hoh[365]), xytext=(365*1.5, acf_hoh[365]+0.05),
                arrowprops=dict(arrowstyle='->', color='red'), fontsize=10, color='red')

# ============================================================================
# If Lamar exists: Comparison plots
# ============================================================================

if has_lamar:
    # Plot 5: Lamar ACF
    ax5 = fig.add_subplot(gs[2, 0])

    lags_plot_lamar = np.arange(len(acf_lamar))
    ax5.plot(lags_plot_lamar, acf_lamar, linewidth=2, color='green', alpha=0.8)
    ax5.axhline(y=1/np.e, color='red', linestyle='--', linewidth=2, label=f'1/e threshold (decorr = {decorr_lamar} days)')
    ax5.axvspan(0, 60, alpha=0.15, color='orange', label='Daily model (60 days)')

    ax5.set_xlabel('Lag (days)', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Autocorrelation', fontsize=12, fontweight='bold')
    ax5.set_title('Lamar River: Autocorrelation Function', fontsize=14, fontweight='bold')
    ax5.set_xlim(0, min(365*3, len(acf_lamar)))
    ax5.grid(True, alpha=0.3)
    ax5.legend(fontsize=9)

    # Plot 6: Direct comparison
    ax6 = fig.add_subplot(gs[2, 1])

    max_compare = min(365*2, len(acf_hoh), len(acf_lamar))
    compare_lags = np.arange(max_compare)

    ax6.plot(compare_lags, acf_hoh[:max_compare], linewidth=2, color='blue', alpha=0.8, label=f'Hoh (H={H_hoh:.3f})')
    ax6.plot(compare_lags, acf_lamar[:max_compare], linewidth=2, color='green', alpha=0.8, label=f'Lamar (H={H_lamar:.3f})')
    ax6.axvline(x=60, color='orange', linestyle='--', linewidth=2, alpha=0.5, label='Model lookback')

    ax6.set_xlabel('Lag (days)', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Autocorrelation', fontsize=12, fontweight='bold')
    ax6.set_title('Hoh vs Lamar: Joseph Effect Comparison', fontsize=14, fontweight='bold')
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)

plt.suptitle('Joseph Effect Analysis: Can Models Learn Long-Range Persistence?',
             fontsize=16, fontweight='bold', y=0.995)

plt.savefig('../data/joseph_effect_analysis.png', dpi=150, bbox_inches='tight')
print("\n✓ Visualization saved to: ../data/joseph_effect_analysis.png")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 80)
print("KEY FINDINGS")
print("=" * 80)

print(f"\n1. HOH RIVER:")
print(f"   • Hurst exponent H = {H_hoh:.3f}")
print(f"   • Decorrelation time = {decorr_hoh} days ({decorr_hoh/365.25:.1f} years)")
print(f"   • Lag-1 autocorrelation = {acf_hoh[1]:.3f}")
print(f"   • Lag-365 autocorrelation = {acf_hoh[365]:.3f}")

if has_lamar:
    print(f"\n2. LAMAR RIVER:")
    print(f"   • Hurst exponent H = {H_lamar:.3f}")
    print(f"   • Decorrelation time = {decorr_lamar} days ({decorr_lamar/365.25:.1f} years)")
    print(f"   • Lag-1 autocorrelation = {acf_lamar[1]:.3f}")
    print(f"   • Lag-365 autocorrelation = {acf_lamar[365]:.3f}")

print(f"\n3. MODEL LIMITATIONS:")
print(f"   • Daily model SEQ_LENGTH = 60 days")
print(f"   • Hourly model SEQ_LENGTH = 7 days")
print(f"   • Joseph Effect operates on YEARS (decorr = {decorr_hoh/365.25:.1f} years for Hoh)")
print(f"   • ⚠️ Models CANNOT learn multi-year persistence directly")

print(f"\n4. WHAT MODELS ACTUALLY LEARN:")
print(f"   ✓ Short-term autocorrelation (lags 1-60 days)")
print(f"   ✓ 'Tomorrow ≈ today' works because of high lag-1 correlation")
print(f"   ✗ Multi-year wet/dry cycles (Joseph Effect)")
print(f"   ✗ Decadal climate oscillations (PDO, etc.)")

print(f"\n5. WHY THIS MATTERS:")
print(f"   • High R² comes from short-term persistence, NOT long-term cycles")
if has_lamar:
    print(f"   • Lamar gets R² ≈ 0.97 because lag-1 r = {acf_lamar[1]:.3f}")
print(f"   • Hoh gets R² ≈ 0.66 because lag-1 r = {acf_hoh[1]:.3f}")
print(f"   • Neither model learns the Joseph Effect itself")

print(f"\n6. TRAINING DATA IMPLICATIONS:")
print(f"   • 10 years vs 36 years matters for REGIME DIVERSITY")
print(f"   • NOT because model learns multi-year cycles")
print(f"   • BUT because it sees wet AND dry conditions")
print(f"   • Generalization requires diverse training regimes")

print("\n" + "=" * 80)
