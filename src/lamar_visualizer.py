import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set style for professional hydrological plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

def plot_horizon_performance(dates, actuals, forecasts, horizon_indices=[0, 2, 6]):
    """
    Plots the full time series comparing Actuals vs Forecasts for specific horizons.
    
    Args:
        dates: Array of dates corresponding to the START of the forecast window.
        actuals: Array of shape (N,) containing true flow values.
                 NOTE: Must be aligned so actuals[i] is the target for forecasts[i, 0]
        forecasts: Array of shape (N, 7) containing predicted sequences.
        horizon_indices: List of indices to plot (0=Day 1, 2=Day 3, 6=Day 7).
    """
    plt.figure(figsize=(14, 7))
    
    # Plot Actuals (Black solid line)
    # We align actuals to the dates. 
    # If forecasts[i, 0] is prediction for date[i]+1day, we plot accordingly.
    
    # We assume 'actuals' passed here aligns with 'dates' + 1 day (the first target)
    plt.plot(dates, actuals, color='black', linewidth=2, label='Actual Flow', alpha=0.8)
    
    colors = ['#2ecc71', '#3498db', '#e74c3c'] # Green, Blue, Red
    
    for i, h_idx in enumerate(horizon_indices):
        # The forecast for horizon h (e.g., 3 days out) was made h days ago.
        # We need to shift the plot so the prediction lines up with the valid date.
        # Forecast[t, h] is a prediction for time t + h + 1
        
        valid_dates = dates + pd.Timedelta(days=h_idx + 1)
        pred_series = forecasts[:, h_idx]
        
        label = f'Day {h_idx+1} Forecast'
        plt.plot(valid_dates, pred_series, color=colors[i], linestyle='--', 
                 linewidth=1.5, label=label, alpha=0.9)

    plt.title('Lamar River: Actual vs. Forecasts by Lead Time', fontsize=16)
    plt.ylabel('Discharge (cfs)', fontsize=12)
    plt.xlabel('Date', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_spaghetti_event(dates, actuals, forecasts, start_date, end_date):
    """
    Plots a "Spaghetti" plot for a specific event window.
    Shows the 7-day trajectory launched from every day in the window.
    
    Args:
        dates: Array of dates corresponding to the forecast launch date.
        actuals: Full array of actuals (aligned to dates + 1).
        forecasts: Full array of forecasts (N, 7).
        start_date: Str, start of zoom window (e.g., '2022-06-01').
        end_date: Str, end of zoom window.
    """
    # Filter indices
    mask = (dates >= start_date) & (dates <= end_date)
    subset_dates = dates[mask]
    subset_forecasts = forecasts[mask]
    
    # Get actuals for a slightly wider window to show context
    plot_start = pd.to_datetime(start_date)
    plot_end = pd.to_datetime(end_date) + pd.Timedelta(days=10)
    
    # Find indices for actuals
    # Assuming actuals array is 1-to-1 with dates array (shifted by 1 day)
    act_mask = (dates >= start_date) & (dates <= end_date) # Simplified alignment for demo
    subset_actuals = actuals[act_mask]
    
    plt.figure(figsize=(14, 8))
    
    # Plot 1: The Base Actuals
    # We reconstruct the 'valid date' for the actuals
    valid_dates_act = subset_dates + pd.Timedelta(days=1)
    plt.plot(valid_dates_act, subset_actuals, color='black', linewidth=3, 
             label='Observed Flow', zorder=10)
    
    # Plot 2: The Spaghetti Strands
    # Each row in subset_forecasts is a 7-day sequence
    for i in range(len(subset_dates)):
        launch_date = subset_dates[i]
        # X-axis for this specific strand: [launch+1, ..., launch+7]
        strand_dates = [launch_date + pd.Timedelta(days=d+1) for d in range(7)]
        strand_values = subset_forecasts[i]
        
        # Color fades from Red (Day 1) to Yellow (Day 7) or just use a single color
        plt.plot(strand_dates, strand_values, color='#3498db', alpha=0.4, linewidth=1.5, marker='o', markersize=3)

    plt.title(f'Forecast "Spaghetti" Plot: {start_date} to {end_date}', fontsize=16)
    plt.ylabel('Discharge (cfs)', fontsize=12)
    plt.xlabel('Date', fontsize=12)
    
    # Create custom legend element for the strands
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='black', lw=3),
                    Line2D([0], [0], color='#3498db', lw=1.5, alpha=0.8)]
    plt.legend(custom_lines, ['Observed', '7-Day Forecast Trails'])
    
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.show()

def plot_error_decay(actuals, forecasts):
    """
    Calculates and plots NSE/RMSE for each of the 7 lead times.
    """
    horizons = np.arange(1, 8)
    nse_scores = []
    
    for i in range(7):
        # Forecast column i corresponds to horizon i+1
        pred = forecasts[:, i]
        # Actuals need to be shifted? 
        # If forecasts[t, 0] is t+1, and actuals[t] is t+1 (aligned)
        # We assume for this function they are pre-aligned or we use simple subtraction
        # For simplicity in this demo, we assume simple alignment
        
        # In real world: You must carefully align arrays. 
        # Here we just compute metrics on the raw passed arrays assuming alignment.
        
        numerator = np.sum((actuals - pred) ** 2)
        denominator = np.sum((actuals - np.mean(actuals)) ** 2)
        nse = 1 - (numerator / denominator)
        nse_scores.append(nse)
        
    plt.figure(figsize=(8, 5))
    plt.plot(horizons, nse_scores, marker='o', linestyle='-', color='#8e44ad', linewidth=2)
    plt.title('Model Skill Degradation (NSE vs Lead Time)', fontsize=14)
    plt.xlabel('Forecast Horizon (Days)', fontsize=12)
    plt.ylabel('Nash-Sutcliffe Efficiency (NSE)', fontsize=12)
    plt.ylim(0, 1.0)
    plt.grid(True)
    plt.show()


# ==========================================
# MOCK DATA DEMO (To verify the plots)
# ==========================================
if __name__ == "__main__":
    print("Generating mock data for visualization...")
    
    # 1. Create a timeline
    dates = pd.date_range(start='2022-04-01', end='2022-08-01', freq='D')
    N = len(dates)
    
    # 2. Create synthetic 'Actual' flow (Gaussian peak for snowmelt)
    days = np.arange(N)
    peak_day = 60 # June 1st approx
    actuals = 500 + 4000 * np.exp(-((days - peak_day)**2) / (2 * 15**2)) 
    actuals += np.random.normal(0, 100, N) # Add noise
    
    # 3. Create synthetic 'Forecasts' (N samples, 7 horizons)
    # The model predicts the actuals but with increasing error/lag as horizon increases
    forecasts = np.zeros((N, 7))
    
    for i in range(N):
        for h in range(7): # h=0 is 1 day out, h=6 is 7 days out
            target_idx = i + h 
            if target_idx < N:
                # Perfect forecast + increasing noise + slight dampening (regression to mean)
                noise_scale = 50 * (h + 1) 
                pred = actuals[target_idx] + np.random.normal(0, noise_scale)
                forecasts[i, h] = pred
            else:
                forecasts[i, h] = actuals[-1] # Pad end

    print("Displaying Horizon Plot...")
    plot_horizon_performance(dates, actuals, forecasts)
    
    print("Displaying Spaghetti Plot (Runoff Peak)...")
    plot_spaghetti_event(dates, actuals, forecasts, '2022-05-15', '2022-06-15')
    
    print("Displaying Error Decay...")
    plot_error_decay(actuals, forecasts)
