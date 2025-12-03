#!/usr/bin/env python3
"""
Water Balance Model for Hydrological Analysis

This script calculates daily water balance including PET, AET, soil moisture,
runoff, and snowpack dynamics for weather station data.

Can be used as a CLI tool or imported as a library.
"""

import os
import sys
import argparse
import warnings
import datetime
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass, field
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.style
import matplotlib.patches as patches
from io import BytesIO

warnings.filterwarnings("ignore")

# Constants
PI = np.pi
E = np.e
MISSING_VAL_LIST = [9999, 99999, -99999, '9999', '-99999', '-9999', -9999]


@dataclass
class WaterBalanceConfig:
    """Configuration for water balance model"""
    station: str
    year_start: int
    year_end: int
    station_type: str = 'GHCN'  # GHCN, SNOTEL, RAWS
    pet_type: str = 'hamon'  # hamon, hargreaves, oudin, Penman_Montieth
    max_soil_water: float = 100.0  # mm
    gdd_base: float = 4.4444  # degrees C
    snow_density: float = 0.2  # for GHCN snow depth to SWE conversion
    melt_thresh_temperature: float = -6.0  # degrees C
    melt_factor: float = 0.35
    precip_fraction: float = 0.167
    forgive_snow: bool = True
    completely_ignore: bool = False
    use_heatload: bool = False

    # Station metadata (set after initialization)
    latitude: Optional[float] = None
    elevation: Optional[float] = None
    title: Optional[str] = None


@dataclass
class WaterBalanceResults:
    """Results from water balance calculation"""
    dates: np.ndarray
    year: np.ndarray
    month: np.ndarray
    tmax: np.ndarray
    tmin: np.ndarray
    tmean: np.ndarray
    precip: np.ndarray
    rain: np.ndarray
    daily_snow: np.ndarray
    snowpack: np.ndarray
    melt: np.ndarray
    pet: np.ndarray
    aet: np.ndarray
    deficit: np.ndarray
    soil_water: np.ndarray
    runoff: np.ndarray
    water_input: np.ndarray
    agdd: np.ndarray
    accum_precip: np.ndarray

    def to_dict(self) -> Dict[str, np.ndarray]:
        """Convert to dictionary format"""
        return {
            'dates': self.dates,
            'year': self.year,
            'month': self.month,
            'tmax': self.tmax,
            'tmin': self.tmin,
            't': self.tmean,
            'p': self.precip,
            'rain': self.rain,
            'dailysnow': self.daily_snow,
            'Pack': self.snowpack,
            'melt': self.melt,
            'pet': self.pet,
            'aet': self.aet,
            'd': self.deficit,
            'soilwater': self.soil_water,
            'runoff': self.runoff,
            'w': self.water_input,
            'agddc': self.agdd,
            'accumprecip': self.accum_precip
        }


class SoilReservoir:
    """Manages soil water storage and dynamics"""

    def __init__(self, max_holding_capacity: float):
        self.stored_water = 0.0
        self.max = float(max_holding_capacity)

    def add(self, amount: float) -> float:
        """Add water to soil, return runoff"""
        if not np.isnan(amount):
            self.stored_water += amount
        runoff = max(0, self.stored_water - self.max)
        if runoff > 0:
            self.stored_water = self.max
        return runoff

    def remove_fraction(self, pet: float, w: float) -> float:
        """Remove water from soil based on PET and available water"""
        exponent = ((pet - w) / self.max) * -1
        multiplier = (1.0 - E ** exponent)
        amount_removed = self.stored_water * multiplier

        if not np.isnan(pet) and not np.isnan(w):
            self.stored_water -= amount_removed
            if self.stored_water < 0:
                self.stored_water = 0
        return amount_removed


class PETCalculator:
    """Calculates potential evapotranspiration using various methods"""

    @staticmethod
    def calc_inverse_rel_distance(doy: float) -> float:
        """Calculate inverse relative distance Earth-Sun"""
        bracket_term = ((2 * PI) / 365) * doy
        return (0.033 * np.cos(bracket_term)) + 1

    @staticmethod
    def calc_solar_declination(doy: float) -> float:
        """Calculate solar declination"""
        bracket_term = (((2 * PI) / 365) * doy) - 1.39
        return 0.409 * np.sin(bracket_term)

    @staticmethod
    def calc_sunset_hour_angle(latitude: float, solar_declination: float) -> float:
        """Calculate sunset hour angle"""
        bracket_term = np.tan(latitude) * -1 * np.tan(solar_declination)
        return np.arccos(bracket_term)

    @staticmethod
    def calculate_Ra(inverse_rel_distance: float, sunset_hour_angle: float,
                     latitude: float, solar_declination: float) -> float:
        """Calculate extraterrestrial radiation"""
        Gsc = 0.0820  # Solar constant MJ/m^2/day
        first_term = (float((24 * 60)) / PI) * Gsc
        first_half = (np.sin(latitude) * np.sin(solar_declination)) * sunset_hour_angle
        second_half = (np.cos(latitude) * np.cos(solar_declination) * np.sin(sunset_hour_angle))
        full_bracket = first_half + second_half
        return first_term * full_bracket * inverse_rel_distance

    @staticmethod
    def calc_daylength(sunset_hour_angle: float) -> float:
        """Calculate day length in hours"""
        return (24 / PI) * sunset_hour_angle

    @staticmethod
    def convert_decimal_latitude_to_radians(lat: float) -> float:
        """Convert latitude to radians"""
        return (PI / 180) * lat

    @classmethod
    def hargreaves(cls, tmax: float, tmin: float, Ra: float) -> float:
        """Calculate PET using Hargreaves method"""
        if tmax in MISSING_VAL_LIST or tmin in MISSING_VAL_LIST:
            return np.nan

        tmean = (tmax + tmin) / 2
        trange = tmax - tmin
        sqrt_trange = np.sqrt(trange)
        offset_tmean = tmean + 17.8
        Et = 0.0023 * offset_tmean * sqrt_trange * Ra
        return max(0, Et)

    @classmethod
    def hamon(cls, tmax: float, tmin: float, daytime_length: float) -> float:
        """Calculate PET using Hamon method"""
        es = cls._calc_Es(tmax, tmin)
        tmean = (tmax + tmin) / 2
        bottom = tmean + 273.3
        bracket = es / bottom
        return 29.8 * daytime_length * bracket

    @classmethod
    def oudin(cls, tmax: float, tmin: float, Ra: float) -> float:
        """Calculate PET using Oudin method"""
        tmean = (tmax + tmin) / 2
        top = Ra * (tmean + 5) * 0.408
        bottom = 100
        PET = top / bottom
        return np.where(tmean > -5, PET, 0)

    @staticmethod
    def _calc_Es(tmax: float, tmin: float) -> float:
        """Calculate saturation vapor pressure"""
        def calc_svp(t):
            bracket_term = (17.27 * t) / (t + 237.3)
            return 0.6108 * np.exp(bracket_term)

        e_tmax = calc_svp(tmax)
        e_tmin = calc_svp(tmin)
        return (e_tmax + e_tmin) / 2

    @classmethod
    def calculate_pet(cls, day_of_year: int, tmax: float, tmin: float,
                      latitude: float, elevation: float, pet_type: str) -> float:
        """Calculate PET using specified method"""
        inverse_rel_distance = cls.calc_inverse_rel_distance(day_of_year)
        solar_declination = cls.calc_solar_declination(day_of_year)
        lat_radians = cls.convert_decimal_latitude_to_radians(latitude)
        sunset_hour_angle = cls.calc_sunset_hour_angle(lat_radians, solar_declination)
        Ra = cls.calculate_Ra(inverse_rel_distance, sunset_hour_angle,
                              lat_radians, solar_declination)
        daylength = cls.calc_daylength(sunset_hour_angle)

        if pet_type == 'oudin':
            return cls.oudin(tmax, tmin, Ra)
        elif pet_type == 'hargreaves':
            return cls.hargreaves(tmax, tmin, Ra)
        elif pet_type == 'hamon':
            return cls.hamon(tmax, tmin, daylength)
        else:
            raise ValueError(f"Unknown PET type: {pet_type}")


class SnowModel:
    """Manages snow accumulation and melt"""

    def __init__(self, melt_thresh_temp: float = -6.0, melt_factor: float = 0.35,
                 precip_fraction: float = 0.167):
        self.melt_thresh_temp = melt_thresh_temp
        self.melt_factor = melt_factor
        self.precip_fraction = precip_fraction

    def melt_one_day(self, start_swe: float, tavg: float) -> float:
        """Calculate daily snowmelt using degree-day method"""
        if tavg <= self.melt_thresh_temp:
            return start_swe

        swe_delta = (tavg - self.melt_thresh_temp) * self.melt_factor
        end_swe = start_swe - swe_delta
        return max(0, end_swe)

    def accum_one_day(self, start_swe: float, tavg: float, precip: float) -> float:
        """Calculate daily snow accumulation"""
        if tavg <= self.melt_thresh_temp:
            rain = 0.0
        elif tavg > (self.melt_thresh_temp + 6):
            rain = 1.0
        else:
            rain = self.precip_fraction * (tavg - self.melt_thresh_temp)

        snow_increment = (1.0 - rain) * precip
        snow_increment = max(0, snow_increment)
        return start_swe + snow_increment

    def estimate_snow(self, swe: float, tavg: float, precip: float) -> float:
        """Estimate snow water equivalent for one day"""
        swe = self.melt_one_day(swe, tavg)
        swe = self.accum_one_day(swe, tavg, precip)
        return swe


def convert_F_to_C(tval: float) -> float:
    """Convert Fahrenheit to Celsius"""
    if tval in MISSING_VAL_LIST:
        return np.nan
    return (tval - 32) / 1.8


def convert_inches_to_mm(pval: float) -> float:
    """Convert inches to millimeters"""
    if pval in MISSING_VAL_LIST:
        return np.nan
    return pval * 25.4


def simplified_heat_load(station_latitude: float) -> float:
    """Calculate simplified heat load index"""
    latitude = (PI / 180) * station_latitude
    return 0.339 + 0.808 * np.cos(latitude)


def calc_aet_and_runoff(soil_water: SoilReservoir, w: float, pet: float) -> Tuple[float, float]:
    """Calculate actual evapotranspiration and runoff"""
    if w > pet:
        aet = pet
        excess = w - aet
        runoff = soil_water.add(excess)
    else:
        runoff = 0
        delta_soil_water = soil_water.remove_fraction(pet, w)
        total_w_avail = delta_soil_water + w
        if pet > total_w_avail:
            aet = total_w_avail
        else:
            aet = pet
    return aet, runoff


def run_water_balance(data: Dict, config: WaterBalanceConfig,
                      start_date: datetime.datetime,
                      end_date: datetime.datetime,
                      input_units: str = 'imperial') -> WaterBalanceResults:
    """
    Run water balance model

    Args:
        data: Dictionary with daily weather data (dates, tmax, tmin, precip, etc.)
        config: WaterBalanceConfig object
        start_date: Start date for simulation
        end_date: End date for simulation
        input_units: 'imperial' (F, inches) or 'metric' (C, mm)

    Returns:
        WaterBalanceResults object
    """
    # Initialize
    snow_model = SnowModel(config.melt_thresh_temperature, config.melt_factor,
                          config.precip_fraction)
    pet_calc = PETCalculator()
    soil_water = SoilReservoir(config.max_soil_water)

    if config.use_heatload:
        heatload = simplified_heat_load(config.latitude)
    else:
        heatload = 1.0

    # Storage for results
    results = {
        'dates': [], 'year': [], 'month': [],
        'tmax': [], 'tmin': [], 'tmean': [],
        'precip': [], 'rain': [], 'daily_snow': [],
        'snowpack': [], 'melt': [], 'pet': [],
        'aet': [], 'deficit': [], 'soil_water': [],
        'runoff': [], 'water_input': [], 'agdd': [],
        'accum_precip': []
    }

    # State variables
    agddc = 0.0
    accum_precip = 0.0
    accumsnow = 0.0
    last_accumsnow = np.nan
    one_day = datetime.timedelta(days=1)
    current_date = start_date

    i = 0
    while current_date <= end_date:
        month = current_date.month
        day = current_date.day
        year = current_date.year
        date_str = f"{month:02d}/{day:02d}/{year}"

        # Find matching date in data
        try:
            idx = np.where(data['date'] == date_str)[0][0]
        except (IndexError, KeyError):
            current_date += one_day
            continue

        # Get weather data and convert to metric if needed
        tmax_raw = data.get('tmax', [np.nan] * len(data['date']))[idx]
        tmin_raw = data.get('tmin', [np.nan] * len(data['date']))[idx]
        precip_raw = data.get('precip', [np.nan] * len(data['date']))[idx]

        # Convert to metric (C and mm) if input is imperial
        if input_units == 'imperial':
            tmax = convert_F_to_C(tmax_raw)
            tmin = convert_F_to_C(tmin_raw)
            precip = convert_inches_to_mm(precip_raw)
        else:  # metric
            tmax = tmax_raw
            tmin = tmin_raw
            precip = precip_raw

        # Screen unrealistic values
        if tmax > 75:  # degrees C
            tmax = np.nan
        if tmin > 75:
            tmin = np.nan

        tmean = (tmax + tmin) / 2

        # Reset accumulations
        if month == 1 and day == 1:
            agddc = 0.0
        if month == 10 and day == 1:
            accum_precip = 0.0
        else:
            accum_precip += precip

        # Estimate snow
        accumsnow = snow_model.estimate_snow(accumsnow, tmean, precip)

        # Handle missing snow data
        if np.isnan(accumsnow) and config.forgive_snow:
            accumsnow = last_accumsnow

        # Calculate snow dynamics
        if i == 0:
            melt = 0
            daily_snow = 0
            rain = 0
            w = 0
        else:
            melt = max(0, abs(min(0, accumsnow - last_accumsnow)))
            daily_snow = max(0, accumsnow - last_accumsnow)
            rain = max(0, precip - daily_snow)
            w = rain + melt

        # Calculate GDD
        day_of_year = current_date.timetuple().tm_yday
        gdd = max(0, tmean - config.gdd_base)
        if not np.isnan(gdd):
            agddc += gdd

        # Calculate PET
        pet = pet_calc.calculate_pet(day_of_year, tmax, tmin,
                                     config.latitude, config.elevation,
                                     config.pet_type)
        pet = pet * heatload

        # No evaporation when snow present
        if accumsnow > 0 and not np.isnan(pet):
            pet = 0

        # Calculate AET and runoff
        aet, runoff = calc_aet_and_runoff(soil_water, w, pet)
        deficit = max(0, pet - aet)

        # Store results
        results['dates'].append(date_str)
        results['year'].append(str(year))
        results['month'].append(f"{month:02d}")
        results['tmax'].append(tmax)
        results['tmin'].append(tmin)
        results['tmean'].append(tmean)
        results['precip'].append(precip)
        results['rain'].append(rain)
        results['daily_snow'].append(daily_snow)
        results['snowpack'].append(accumsnow)
        results['melt'].append(melt)
        results['pet'].append(pet)
        results['aet'].append(aet)
        results['deficit'].append(deficit)
        results['soil_water'].append(soil_water.stored_water)
        results['runoff'].append(runoff)
        results['water_input'].append(w)
        results['agdd'].append(agddc)
        results['accum_precip'].append(accum_precip)

        last_accumsnow = accumsnow
        current_date += one_day
        i += 1

    # Convert to numpy arrays
    return WaterBalanceResults(
        dates=np.array(results['dates']),
        year=np.array(results['year']),
        month=np.array(results['month']),
        tmax=np.array(results['tmax']),
        tmin=np.array(results['tmin']),
        tmean=np.array(results['tmean']),
        precip=np.array(results['precip']),
        rain=np.array(results['rain']),
        daily_snow=np.array(results['daily_snow']),
        snowpack=np.array(results['snowpack']),
        melt=np.array(results['melt']),
        pet=np.array(results['pet']),
        aet=np.array(results['aet']),
        deficit=np.array(results['deficit']),
        soil_water=np.array(results['soil_water']),
        runoff=np.array(results['runoff']),
        water_input=np.array(results['water_input']),
        agdd=np.array(results['agdd']),
        accum_precip=np.array(results['accum_precip'])
    )


def save_results_to_csv(results: WaterBalanceResults, output_path: str):
    """Save water balance results to CSV file"""
    import csv

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header
        header = [
            'DATE', 'YEAR', 'MONTH', 'TMAX_C', 'TMIN_C', 'TMEAN_C',
            'PRECIP_MM', 'RAIN_MM', 'DAILY_SNOW_MM', 'SNOWPACK_MM',
            'MELT_MM', 'PET_MM', 'AET_MM', 'DEFICIT_MM',
            'SOIL_WATER_MM', 'RUNOFF_MM', 'WATER_INPUT_MM',
            'AGDD_C', 'ACCUM_PRECIP_MM'
        ]
        writer.writerow(header)

        # Data rows
        for i in range(len(results.dates)):
            row = [
                results.dates[i],
                results.year[i],
                results.month[i],
                f"{results.tmax[i]:.2f}",
                f"{results.tmin[i]:.2f}",
                f"{results.tmean[i]:.2f}",
                f"{results.precip[i]:.2f}",
                f"{results.rain[i]:.2f}",
                f"{results.daily_snow[i]:.2f}",
                f"{results.snowpack[i]:.2f}",
                f"{results.melt[i]:.2f}",
                f"{results.pet[i]:.2f}",
                f"{results.aet[i]:.2f}",
                f"{results.deficit[i]:.2f}",
                f"{results.soil_water[i]:.2f}",
                f"{results.runoff[i]:.2f}",
                f"{results.water_input[i]:.2f}",
                f"{results.agdd[i]:.2f}",
                f"{results.accum_precip[i]:.2f}"
            ]
            writer.writerow(row)


def plot_water_balance(results: WaterBalanceResults, title: str,
                       output_path: Optional[str] = None):
    """
    Create water balance visualization

    Args:
        results: WaterBalanceResults object
        title: Plot title
        output_path: Optional path to save plot (if None, returns figure)
    """
    matplotlib.style.use('classic')
    fig = plt.figure(figsize=(14, 9))

    dates = results.dates
    cats = np.arange(len(dates))

    # Main plot - PET, AET, Deficit
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(cats, results.pet, '-', color='red', linewidth=2, label='PET')
    ax1.plot(cats, results.aet, '-', color='blue', linewidth=1, label='AET')
    ax1.fill_between(cats, results.pet, results.aet,
                     where=results.pet >= results.aet,
                     facecolor='orange', alpha=0.35, label='Deficit')

    ax1.set_ylabel('AET, PET and moisture deficit (mm)')
    ax1.set_title(f"{title} Water Balance")
    ax1.legend(loc='upper left')

    # Secondary axis - Soil water
    ax2 = ax1.twinx()
    ax2.plot(cats, results.soil_water, '-', color='green',
             linewidth=4, alpha=0.7, label='Soil water')
    ax2.set_ylabel('Soil water (mm)')
    ax2.legend(loc='upper right')

    # X-axis labels (sparse)
    label_interval = max(1, len(dates) // 15)
    label_cats = cats[::label_interval]
    labels = dates[::label_interval]
    plt.xticks(label_cats, labels, rotation=90)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        return fig


def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description='Water Balance Model for Hydrological Analysis',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument('--station', required=True,
                       help='Station name or ID')
    parser.add_argument('--year-start', type=int, required=True,
                       help='Start year for analysis')
    parser.add_argument('--year-end', type=int, required=True,
                       help='End year for analysis')
    parser.add_argument('--latitude', type=float, required=True,
                       help='Station latitude (decimal degrees)')
    parser.add_argument('--elevation', type=float, required=True,
                       help='Station elevation (meters)')
    parser.add_argument('--input-file', required=True,
                       help='Path to input weather data file')

    # Optional arguments
    parser.add_argument('--station-type', default='GHCN',
                       choices=['GHCN', 'SNOTEL', 'RAWS'],
                       help='Station type')
    parser.add_argument('--pet-type', default='hamon',
                       choices=['hamon', 'hargreaves', 'oudin'],
                       help='PET calculation method')
    parser.add_argument('--max-soil-water', type=float, default=100.0,
                       help='Maximum soil water capacity (mm)')
    parser.add_argument('--gdd-base', type=float, default=4.4444,
                       help='Base temperature for growing degree days (C)')
    parser.add_argument('--use-heatload', action='store_true',
                       help='Apply latitude-based heat load correction to PET')
    parser.add_argument('--forgive-snow', action='store_true',
                       help='Fill missing snow values with previous day')
    parser.add_argument('--output-csv', required=True,
                       help='Output CSV file path')
    parser.add_argument('--output-plot',
                       help='Output plot file path (optional)')
    parser.add_argument('--title',
                       help='Title for output plot')

    args = parser.parse_args()

    # Create configuration
    config = WaterBalanceConfig(
        station=args.station,
        year_start=args.year_start,
        year_end=args.year_end,
        station_type=args.station_type,
        pet_type=args.pet_type,
        max_soil_water=args.max_soil_water,
        gdd_base=args.gdd_base,
        use_heatload=args.use_heatload,
        forgive_snow=args.forgive_snow,
        latitude=args.latitude,
        elevation=args.elevation,
        title=args.title or args.station
    )

    # Load input data (this would need to be implemented based on your data format)
    print(f"Loading data from {args.input_file}...")
    # data = load_weather_data(args.input_file)  # You'll need to implement this

    # For now, provide guidance
    print("\nNOTE: You need to implement data loading for your specific format.")
    print("Expected data dictionary format:")
    print("  data = {")
    print("    'date': ['MM/DD/YYYY', ...],  # dates as strings")
    print("    'tmax': [temp_f, ...],        # max temp in Fahrenheit")
    print("    'tmin': [temp_f, ...],        # min temp in Fahrenheit")
    print("    'precip': [inches, ...],      # precip in inches")
    print("  }")
    print("\nAfter loading data, call:")
    print("  results = run_water_balance(data, config, start_date, end_date)")
    print("  save_results_to_csv(results, args.output_csv)")

    return 0


if __name__ == '__main__':
    sys.exit(main())
