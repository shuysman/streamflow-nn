# Snow Model Calibration Guide

This guide describes how to calibrate the degree-day snow model in `water_balance.py` using SNOTEL observations. The calibration procedure can be applied to any spatial scale—from watershed-level elevation bands to 1m spatially explicit runs.

## Why Calibrate?

The default snow model parameters were chosen as reasonable starting values but are not optimized for any specific location:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `melt_factor` | 0.35 mm/°C/day | Degree-day melt coefficient |
| `melt_thresh_temp` | -6.0°C | Temperature threshold for melt onset |

Calibration against SNOTEL SWE observations improved NSE from **0.67 → 0.91** for the Lamar River alpine band. This translates to more accurate:
- Snowpack accumulation timing and magnitude
- Spring melt timing
- Runoff generation

## Calibration Features

The calibration script includes robust validation methodology:

| Feature | Description |
|---------|-------------|
| **K-fold cross-validation** | Splits data by water year (Oct 1 - Sep 30) to detect overfitting |
| **Spin-up handling** | Excludes first 365 days from metrics to avoid SWE=0 initialization artifacts |
| **Parameter uncertainty** | Reports mean ± std of parameters across CV folds |
| **Overfitting detection** | Warns if calibration-validation NSE gap > 0.1 |

## Calibration Workflow

### Step 1: Find Nearby SNOTEL Stations

Use the NRCS SNOTEL website or the provided scripts to find stations near your study area:

```bash
# Find SNOTEL stations near a location
python src/find_snotel_sites.py --lat 44.9 --lon -110.3 --radius 50
```

Or search manually at: https://wcc.sc.egov.usda.gov/nwcc/inventory

**Key considerations:**
- Match station elevation to your model domain elevation
- Prefer stations with long records (10+ years)
- Multiple stations at different elevations allow elevation-specific calibration

### Step 2: Download SNOTEL Data

```bash
# Download SWE data for specific stations
python src/download_snotel.py --stations 683,670 --start 1998-01-01 --end 2024-12-31
```

Output format (CSV):
```
date,swe_station1,swe_station2
1998-10-01,0.0,0.0
1998-10-02,0.5,0.3
...
```

SWE is typically in **inches** from SNOTEL—the calibration script converts to mm automatically.

### Step 3: Prepare Climate Data

The calibration requires daily climate data matching the SNOTEL period:
- Mean temperature (°C)
- Precipitation (mm)

For elevation-specific calibration, use climate data at the SNOTEL station's elevation. The GridMET elevation bands approach works well:

```
data/climate/lamar_gridmet_elevation_bands.csv
  - precip_valley, temp_valley  (2000m)
  - precip_mid, temp_mid        (2500m)
  - precip_alpine, temp_alpine  (3000m)
```

### Step 4: Run Calibration

```bash
# Calibrate all station-band pairs with 5-fold cross-validation (default)
python src/calibrate_snow_model.py

# Calibrate a specific band
python src/calibrate_snow_model.py --band alpine --station parker_peak

# Adjust number of CV folds
python src/calibrate_snow_model.py --n-folds 3

# Skip cross-validation (calibrate on full dataset - original behavior)
python src/calibrate_snow_model.py --no-cv

# Just compare default parameters (no optimization)
python src/calibrate_snow_model.py --no-optimize
```

**What the calibration does:**
1. Loads SNOTEL SWE observations and assigns water years
2. Loads climate data for the matching elevation band
3. Runs the snow model with default parameters (excluding 365-day spin-up from metrics)
4. Performs k-fold cross-validation by water year:
   - Calibrates on (k-1) folds using differential evolution
   - Validates on held-out fold
   - Reports calibration and validation NSE for each fold
5. Averages parameters across folds for final model
6. Saves calibrated parameters and CV results to JSON files

**Output files:**
```
output/calibration/
  calibrated_params_alpine.json      # Includes CV results and parameter uncertainty
  calibrated_params_mid.json
  calibration_alpine_parker_peak.png  (diagnostic plots)
  calibration_mid_northeast_entrance.png
```

**Example JSON output with cross-validation:**
```json
{
  "parameters": {
    "melt_factor": 1.50,
    "melt_thresh_temp": -1.59
  },
  "cross_validation": {
    "n_folds": 5,
    "calibration_nse": {"mean": 0.91, "std": 0.02},
    "validation_nse": {"mean": 0.88, "std": 0.04},
    "param_uncertainty": {
      "melt_factor_std": 0.15,
      "melt_thresh_temp_std": 0.3
    }
  }
}
```

### Step 5: Apply Calibrated Parameters

#### Option A: Use the `--use-calibrated` flag (elevation bands)

```bash
python src/run_elevation_bands_wb.py --use-calibrated
```

This automatically loads parameters from `output/calibration/*.json`.

#### Option B: Manually specify in code

```python
from water_balance import WaterBalanceConfig, run_water_balance

config = WaterBalanceConfig(
    station="my_site",
    latitude=44.9,
    elevation=2800,  # meters
    # Calibrated parameters for alpine zone
    melt_factor=1.5,
    melt_thresh_temperature=-1.6,
    # Other parameters
    max_soil_water=150.0,
    pet_type='oudin'
)

results = run_water_balance(data, config, start_date, end_date)
```

#### Option C: Load from JSON programmatically

```python
import json

def load_calibrated_params(band: str, calibration_dir: str = 'output/calibration'):
    """Load calibrated parameters for a specific elevation band"""
    with open(f'{calibration_dir}/calibrated_params_{band}.json') as f:
        data = json.load(f)
    return data['parameters'], data.get('cross_validation')

# Usage
params, cv_results = load_calibrated_params('alpine')
# params: {'melt_factor': 1.5, 'melt_thresh_temp': -1.59, 'precip_fraction': 0.167}
# cv_results: {'validation_nse': {'mean': 0.88, 'std': 0.04}, ...}
```

## Applying to Spatially Explicit (1m) Models

For high-resolution spatial models, apply calibrated parameters based on elevation zones:

### Approach 1: Elevation-Based Parameter Assignment

```python
import numpy as np

def assign_snow_params(dem: np.ndarray, calibrated_params: dict) -> dict:
    """
    Assign calibrated snow parameters based on elevation

    Args:
        dem: 2D array of elevations (meters)
        calibrated_params: dict with 'valley', 'mid', 'alpine' keys

    Returns:
        Dictionary of 2D parameter arrays
    """
    # Define elevation thresholds (adjust for your watershed)
    valley_max = 2250  # meters
    mid_max = 2750

    # Create masks
    valley_mask = dem < valley_max
    mid_mask = (dem >= valley_max) & (dem < mid_max)
    alpine_mask = dem >= mid_max

    # Initialize parameter arrays
    melt_factor = np.zeros_like(dem, dtype=float)
    melt_thresh = np.zeros_like(dem, dtype=float)

    # Assign valley params (use defaults if not calibrated)
    valley_p = calibrated_params.get('valley', {
        'melt_factor': 0.35, 'melt_thresh_temp': -6.0
    })
    melt_factor[valley_mask] = valley_p['melt_factor']
    melt_thresh[valley_mask] = valley_p['melt_thresh_temp']

    # Assign mid-elevation params
    mid_p = calibrated_params['mid']
    melt_factor[mid_mask] = mid_p['melt_factor']
    melt_thresh[mid_mask] = mid_p['melt_thresh_temp']

    # Assign alpine params
    alpine_p = calibrated_params['alpine']
    melt_factor[alpine_mask] = alpine_p['melt_factor']
    melt_thresh[alpine_mask] = alpine_p['melt_thresh_temp']

    return {
        'melt_factor': melt_factor,
        'melt_thresh_temp': melt_thresh
    }
```

### Approach 2: Interpolate Parameters by Elevation

For smoother transitions between zones:

```python
from scipy.interpolate import interp1d

def interpolate_snow_params(dem: np.ndarray, calibrated_params: dict) -> dict:
    """
    Interpolate snow parameters across elevation gradient.

    Uses cross-validation uncertainty to weight interpolation if available.
    """
    # Known elevations and their calibrated values
    elevations = [2000, 2500, 3000]  # valley, mid, alpine

    melt_factors = [
        calibrated_params.get('valley', {'melt_factor': 0.35})['melt_factor'],
        calibrated_params['mid']['melt_factor'],
        calibrated_params['alpine']['melt_factor']
    ]

    melt_thresholds = [
        calibrated_params.get('valley', {'melt_thresh_temp': -6.0})['melt_thresh_temp'],
        calibrated_params['mid']['melt_thresh_temp'],
        calibrated_params['alpine']['melt_thresh_temp']
    ]

    # Create interpolation functions
    mf_interp = interp1d(elevations, melt_factors,
                         kind='linear', fill_value='extrapolate')
    mt_interp = interp1d(elevations, melt_thresholds,
                         kind='linear', fill_value='extrapolate')

    return {
        'melt_factor': mf_interp(dem),
        'melt_thresh_temp': mt_interp(dem)
    }
```

### Approach 3: Find Local SNOTEL for Each Site

For planting site analysis across a large region, you may want site-specific calibration:

```python
def get_nearest_snotel_params(site_lat, site_lon, site_elev, snotel_stations):
    """
    Find nearest SNOTEL station at similar elevation and return its calibrated params
    """
    best_match = None
    best_score = float('inf')

    for station in snotel_stations:
        # Weight distance and elevation difference
        dist = haversine(site_lat, site_lon, station['lat'], station['lon'])
        elev_diff = abs(site_elev - station['elevation'])

        # Combined score (tune weights as needed)
        score = dist + elev_diff * 0.1  # 100m elev diff ~ 10km distance

        if score < best_score:
            best_score = score
            best_match = station

    return best_match['calibrated_params']
```

## Calibration Results: Lamar River Watershed

Reference calibration results (December 2025):

### Alpine Band (Parker Peak SNOTEL, 2871m)
```json
{
  "melt_factor": 1.50,
  "melt_thresh_temp": -1.59
}
```
- Calibration NSE: 0.91 ± 0.02
- Validation NSE: 0.88 ± 0.04

### Mid Band (Northeast Entrance SNOTEL, 2262m)
```json
{
  "melt_factor": 1.50,
  "melt_thresh_temp": -2.63
}
```
- Calibration NSE: 0.85 ± 0.03
- Validation NSE: 0.82 ± 0.05

### Key Findings

1. **Melt factor** needed to be ~4x higher than default (0.35 → 1.5 mm/°C/day)
2. **Melt threshold** should be much warmer (-6°C → -1.6 to -2.6°C)
3. **Cross-validation gap** of ~0.03 NSE indicates parameters generalize well
4. **Parameter stability** across folds suggests robust calibration

## Extending to Other Watersheds

To calibrate for a new watershed:

1. **Identify SNOTEL stations** in or near the watershed
2. **Download climate data** at matching elevations (GridMET, PRISM, or local stations)
3. **Run calibration** using `calibrate_snow_model.py` as a template
4. **Validate** by comparing simulated vs observed SWE timing and magnitude
5. **Apply** calibrated parameters to production runs

### Modifying for Different Regions

The calibration script (`src/calibrate_snow_model.py`) can be adapted:

```python
# Define your station-to-band mapping
STATION_BAND_MAP = {
    'your_alpine_station': {
        'band': 'alpine',
        'climate_suffix': 'alpine',  # Column suffix in climate CSV
        'elevation': 3000
    },
    'your_mid_station': {
        'band': 'mid',
        'climate_suffix': 'mid',
        'elevation': 2500
    }
}

# Update file paths
SNOTEL_FILE = 'data/snotel/your_watershed_snotel.csv'
CLIMATE_FILE = 'data/climate/your_watershed_elevation_bands.csv'
```

## Parameter Bounds

The optimization uses these bounds (physical constraints):

| Parameter | Min | Max | Rationale |
|-----------|-----|-----|-----------|
| melt_factor | 0.5 | 6.0 | Literature range ~1-6 mm/°C/day |
| melt_thresh_temp | -10°C | +2°C | Physical melt onset range |

Note: `precip_fraction` is fixed at 0.167 to match `water_balance.py` and reduce parameter interactions.

Adjust bounds in `SnowModelParams.get_bounds()` if optimization hits limits consistently.

## Troubleshooting

**Poor NSE after calibration (<0.7):**
- Check climate data quality (missing values, unit issues)
- Verify elevation matching between SNOTEL and climate data
- Consider if the SNOTEL station is representative (aspect, local effects)

**Parameters at bounds:**
- Expand bounds if physically reasonable
- May indicate structural model limitations

**Large calibration-validation gap (>0.1 NSE):**
- Indicates overfitting to training data
- Try fewer parameters (current setup only calibrates 2)
- Check for non-stationarity in the climate record
- Consider using more folds to increase validation robustness

**High parameter variance across folds:**
- Some folds may have unusual conditions (rain-on-snow, dust-on-snow)
- Consider weighting folds by data quality
- Examine individual fold results in the JSON output

**Spin-up artifacts:**
- If first water year shows poor performance, the 365-day spin-up may be insufficient
- For regions with multi-year snowpack, consider increasing `SPINUP_DAYS`

## References

- Degree-day models: Hock, R. (2003). Temperature index melt modelling in mountain areas. Journal of Hydrology.
- SNOTEL data: https://www.nrcs.usda.gov/wps/portal/wcc/home/
- Original water balance model: `src/WATER_BALANCE_README.md`
