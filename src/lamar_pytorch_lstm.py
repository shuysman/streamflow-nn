import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import math
import sys

# Import the visualizer (must be in the same directory)
try:
    import lamar_visualizer
except ImportError:
    print("Warning: 'lamar_visualizer.py' not found. Visualization steps will be skipped.")

# ==========================================
# 1. CONFIGURATION
# ==========================================
CONFIG = {
    'csv_path': '../data/lamar_river_streamflow.csv',
    'lookback': 60,         # Past 60 days of data
    'horizon': 7,           # Predict next 7 days
    'hidden_dim': 64,       # LSTM hidden units
    'num_layers': 2,        # Stacked LSTM layers
    'dropout': 0.2,
    'batch_size': 32,
    'epochs': 50,
    'learning_rate': 0.001,
    'train_split_date': '2018-01-01' # Training up to 2018, Test 2018-2025
}

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ==========================================
# 2. DATA LOADING (REAL DATA)
# ==========================================
def load_lamar_data(filepath):
    """
    Loads the uploaded Lamar River CSV file.
    Expects columns: 'date', 'streamflow_cfs'
    """
    try:
        df = pd.read_csv(filepath)
        
        # Parse Dates (Handle ISO format with timezone)
        df['date'] = pd.to_datetime(df['date'])
        
        # Remove timezone info to simplify plotting/splitting (make naive)
        df['date'] = df['date'].dt.tz_localize(None)
        
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)
        
        # Create day_of_year for seasonality features
        df['day_of_year'] = df.index.dayofyear
        
        # Ensure target column exists and rename to generic 'discharge' for internal consistency
        if 'streamflow_cfs' in df.columns:
            df['discharge'] = df['streamflow_cfs']
        elif 'streamflow' in df.columns:
             # Fallback if cfs not found, assuming streamflow is the target
             df['discharge'] = df['streamflow']
        else:
            raise ValueError("CSV must contain 'streamflow_cfs' or 'streamflow' column")

        # Handle missing values (Linear interpolation for small gaps)
        df['discharge'] = df['discharge'].interpolate(method='linear')
        
        print(f"Data Loaded: {len(df)} records from {df.index.min().date()} to {df.index.max().date()}")
        return df
        
    except FileNotFoundError:
        print(f"ERROR: File '{filepath}' not found. Please ensure it is in the same directory.")
        sys.exit(1)

# ==========================================
# 3. PREPROCESSING
# ==========================================
def preprocess_data(df):
    data = df.copy()
    
    # A. Feature Engineering: Cyclical Time
    data['sin_doy'] = np.sin(2 * np.pi * data['day_of_year'] / 365.0)
    data['cos_doy'] = np.cos(2 * np.pi * data['day_of_year'] / 365.0)
    
    # B. Log Transform Flow (Stabilize Variance)
    # Adding 1 to avoid log(0)
    data['log_discharge'] = np.log1p(data['discharge'])
    
    # C. Scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    feature_cols = ['log_discharge', 'sin_doy', 'cos_doy']
    data[feature_cols] = scaler.fit_transform(data[feature_cols])
    
    return data, scaler

# ==========================================
# 4. PYTORCH DATASET
# ==========================================
class HydroDataset(Dataset):
    def __init__(self, data, lookback, horizon):
        """
        data: Numpy array of shape (Samples, Features)
              Column 0 MUST be the target (log_discharge)
        """
        self.data = torch.FloatTensor(data)
        self.lookback = lookback
        self.horizon = horizon
        
    def __len__(self):
        return len(self.data) - self.lookback - self.horizon + 1

    def __getitem__(self, idx):
        # Input: Sequence from idx to idx+lookback
        x = self.data[idx : idx + self.lookback]
        # Target: Sequence from idx+lookback to idx+lookback+horizon (Target col 0 only)
        y = self.data[idx + self.lookback : idx + self.lookback + self.horizon, 0]
        return x, y

# ==========================================
# 5. MODEL ARCHITECTURE
# ==========================================
class RiverLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout):
        super(RiverLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# ==========================================
# 6. MAIN EXECUTION PIPELINE
# ==========================================
def main():
    # 1. Load Data
    print("Loading Data...")
    df_raw = load_lamar_data(CONFIG['csv_path'])
    
    # 2. Preprocess
    df_processed, scaler = preprocess_data(df_raw)
    
    # Split into Train/Test (Strict Time Split)
    train_mask = df_processed.index < CONFIG['train_split_date']
    train_df = df_processed[train_mask]
    test_df = df_processed[~train_mask]
    
    print(f"Training Set: {len(train_df)} days")
    print(f"Test Set: {len(test_df)} days")
    
    # Prepare Numpy arrays
    feature_cols = ['log_discharge', 'sin_doy', 'cos_doy']
    train_data = train_df[feature_cols].values
    test_data = test_df[feature_cols].values
    
    # 3. Create Datasets & Loaders
    train_dataset = HydroDataset(train_data, CONFIG['lookback'], CONFIG['horizon'])
    test_dataset = HydroDataset(test_data, CONFIG['lookback'], CONFIG['horizon'])
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    
    # 4. Initialize Model
    model = RiverLSTM(
        input_dim=len(feature_cols),
        hidden_dim=CONFIG['hidden_dim'],
        num_layers=CONFIG['num_layers'],
        output_dim=CONFIG['horizon'],
        dropout=CONFIG['dropout']
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    
    # 5. Training Loop
    print("\nStarting Training...")
    for epoch in range(CONFIG['epochs']):
        model.train()
        train_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_val, y_val in test_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                val_outputs = model(x_val)
                val_loss += criterion(val_outputs, y_val).item()
        
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{CONFIG['epochs']} | Train Loss: {train_loss/len(train_loader):.5f} | Val Loss: {val_loss/len(test_loader):.5f}")

    # 6. Evaluation & Inference
    print("\nRunning Inference on Test Set...")
    model.eval()
    all_preds_scaled = []
    
    with torch.no_grad():
        for x_val, y_val in test_loader:
            x_val = x_val.to(device)
            preds = model(x_val).cpu().numpy()
            all_preds_scaled.append(preds)
            
    # Concatenate all predictions
    all_preds_scaled = np.vstack(all_preds_scaled)
    
    # 6b. Prepare Data for Visualization
    # We need to align the predictions with the dates.
    # The Test Dataset starts at index 0 of test_data. 
    # The first input (idx=0) corresponds to dates[0:lookback].
    # The prediction is for dates[lookback : lookback+horizon].
    # So the 'forecast launch date' for prediction i is test_df.index[lookback + i - 1]
    
    # Let's get the dates corresponding to the START of the forecast window (Launch Date)
    # The inputs end at index `lookback-1`. The prediction starts at `lookback`.
    # So the prediction is made "on" date `lookback-1`.
    
    start_idx = CONFIG['lookback'] - 1
    end_idx = start_idx + len(all_preds_scaled)
    
    # Get the valid dates for the test set
    # Note: len(dataset) = len(data) - lookback - horizon + 1
    # We grab the dates corresponding to the day BEFORE the forecast starts (the day we make the pred)
    forecast_dates = test_df.index[start_idx : end_idx]
    
    # 6c. Inverse Transform
    # We need to inverse transform both predictions and the full actual test set
    def inverse_transform_target(scaled_targets, scaler_obj):
        # Create dummy array with shape (N, 3) to satisfy scaler
        dummy = np.zeros((scaled_targets.size, 3))
        dummy[:, 0] = scaled_targets.flatten()
        inverse = scaler_obj.inverse_transform(dummy)[:, 0]
        # Reverse Log Transform: exp(x) - 1
        return np.expm1(inverse).reshape(scaled_targets.shape)

    # Transform Predictions
    real_preds = inverse_transform_target(all_preds_scaled, scaler)
    
    # Transform Actuals (Full test set)
    # We pull the raw 'discharge' column from test_df to ensure 100% alignment
    # But we need to un-log it first because we logged it in preprocessing but didn't scale the df in place?
    # Wait, preprocess_data returns a NEW dataframe with scaled cols.
    # Let's just use the inverse transform on the scaled test_data to be safe.
    real_actuals_full = inverse_transform_target(test_data[:, 0], scaler)
    
    # We need to trim real_actuals_full to match the alignment if we want 1-to-1 array, 
    # but the visualizer takes the full array and dates and handles alignment internally.
    # The visualizer expects 'actuals' to be aligned such that actuals[i] is the target for date[i]+1.
    # Our 'forecast_dates' are the launch dates.
    # The prediction real_preds[i, 0] is for forecast_dates[i] + 1 day.
    # So we need actuals where actuals[i] corresponds to forecast_dates[i] + 1 day.
    
    # Let's simplify: pass the FULL test actuals and the FULL test dates, 
    # but we need to slice them to match the predictions for the visualizer?
    # Actually, lamar_visualizer.plot_horizon_performance takes (dates, actuals, forecasts).
    # It assumes dates[i] is launch date.
    # It assumes actuals[i] is flow at launch date + 1.
    
    # So we need to slice actuals starting from lookback
    aligned_actuals = real_actuals_full[start_idx+1 : end_idx+1]
    
    # If lengths slightly mismatch due to end-of-series boundary, trim to min length
    min_len = min(len(forecast_dates), len(aligned_actuals), len(real_preds))
    forecast_dates = forecast_dates[:min_len]
    aligned_actuals = aligned_actuals[:min_len]
    real_preds = real_preds[:min_len]

    # ==========================================
    # 7. VISUALIZATION
    # ==========================================
    if 'lamar_visualizer' in sys.modules:
        print("\nGenerating Visualizations...")
        
        # A. Performance Overview
        print("1. Horizon Performance Plot")
        lamar_visualizer.plot_horizon_performance(forecast_dates, aligned_actuals, real_preds)
        
        # B. Spaghetti Plot (Peak Runoff Season 2022)
        print("2. Spaghetti Plot (Spring 2022 Runoff)")
        lamar_visualizer.plot_spaghetti_event(forecast_dates, aligned_actuals, real_preds, '2022-05-01', '2022-07-15')
        
        # C. Error Decay
        print("3. Error Decay Curve")
        lamar_visualizer.plot_error_decay(aligned_actuals, real_preds)
    else:
        print("Skipping visualization (module not imported).")

if __name__ == "__main__":
    main()
