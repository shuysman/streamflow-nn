#!/usr/bin/env python3
"""
Cross-validate our water_balance.py against original wb_ai_train.py equations

This manually implements the original equations and compares them
with our refactored versions.
"""

import sys
import numpy as np
from pathlib import Path

# Import our refactored version
from water_balance import PETCalculator, SnowModel

# Manual implementation of original equations for comparison
PI = np.pi
E = np.e


def compare_pet_calculations():
    """Compare PET calculations with original equations"""
    print("="*70)
    print("Comparing PET Calculations with Original Equations")
    print("="*70)

    # Test parameters
    day_of_year = 180  # June 29
    tmax_c = 20.0
    tmin_c = 5.0
    latitude = 44.9
    elevation = 2000.0

    print(f"\nTest parameters:")
    print(f"  Day of year: {day_of_year}")
    print(f"  Tmax: {tmax_c}°C")
    print(f"  Tmin: {tmin_c}°C")
    print(f"  Latitude: {latitude}°")
    print(f"  Elevation: {elevation}m")

    # Our refactored version
    pet_calc = PETCalculator()

    # Calculate intermediate values - OUR VERSION
    inverse_rel_dist_new = pet_calc.calc_inverse_rel_distance(day_of_year)
    solar_decl_new = pet_calc.calc_solar_declination(day_of_year)
    lat_rad_new = pet_calc.convert_decimal_latitude_to_radians(latitude)
    sunset_angle_new = pet_calc.calc_sunset_hour_angle(lat_rad_new, solar_decl_new)
    Ra_new = pet_calc.calculate_Ra(inverse_rel_dist_new, sunset_angle_new,
                                    lat_rad_new, solar_decl_new)
    daylength_new = pet_calc.calc_daylength(sunset_angle_new)

    # Calculate intermediate values - ORIGINAL EQUATIONS (lines 286-330 in wb_ai_train.py)
    # calc_inverse_rel_distance (line 286)
    bracket_term = ((2*PI)/365) * day_of_year
    inverse_rel_dist_orig = (0.033 * np.cos(bracket_term)) + 1

    # calc_solar_declination (line 292)
    bracket_term = (((2*PI)/365) * day_of_year) - 1.39
    solar_decl_orig = 0.409 * np.sin(bracket_term)

    # convert_decimal_latitude_to_radians (line 260)
    lat_rad_orig = (PI/180) * latitude

    # calc_sunset_hour_angle (line 298)
    bracket_term = np.tan(lat_rad_orig) * -1 * np.tan(solar_decl_orig)
    sunset_angle_orig = np.arccos(bracket_term)

    # calculate_Ra (line 201)
    Gsc = 0.0820
    first_term = (float((24*60))/PI) * Gsc
    first_half = (np.sin(lat_rad_orig) * np.sin(solar_decl_orig)) * sunset_angle_orig
    second_half = (np.cos(lat_rad_orig) * np.cos(solar_decl_orig) * np.sin(sunset_angle_orig))
    full_bracket = first_half + second_half
    Ra_orig = first_term * full_bracket * inverse_rel_dist_orig

    # calc_daylength (line 304)
    daylength_orig = (24/PI)*sunset_angle_orig

    print("\n" + "-"*70)
    print("Intermediate calculations comparison:")
    print("-"*70)
    print(f"{'Parameter':<30} {'Original':<15} {'Refactored':<15} {'Diff':<10}")
    print("-"*70)

    comparisons = [
        ("Inverse rel distance", inverse_rel_dist_orig, inverse_rel_dist_new),
        ("Solar declination (rad)", solar_decl_orig, solar_decl_new),
        ("Latitude (rad)", lat_rad_orig, lat_rad_new),
        ("Sunset hour angle (rad)", sunset_angle_orig, sunset_angle_new),
        ("Ra (MJ/m²/day)", Ra_orig, Ra_new),
        ("Daylength (hours)", daylength_orig, daylength_new)
    ]

    all_match = True
    for name, orig_val, new_val in comparisons:
        diff = abs(orig_val - new_val)
        match = "✓" if diff < 0.001 else "✗"
        print(f"{name:<30} {orig_val:<15.6f} {new_val:<15.6f} {diff:<10.6f} {match}")
        if diff >= 0.001:
            all_match = False

    # Test PET methods - ORIGINAL EQUATIONS
    print("\n" + "-"*70)
    print("PET method comparison:")
    print("-"*70)
    print(f"{'Method':<20} {'Original':<15} {'Refactored':<15} {'Diff':<10}")
    print("-"*70)

    # Hargreaves (line 265)
    tmean = (tmax_c + tmin_c) / 2
    trange = tmax_c - tmin_c
    sqrt_trange = np.sqrt(trange)
    offset_tmean = tmean + 17.8
    hargreaves_orig = 0.0023 * offset_tmean * sqrt_trange * Ra_orig
    hargreaves_new = pet_calc.hargreaves(tmax_c, tmin_c, Ra_new)
    diff = abs(hargreaves_orig - hargreaves_new)
    match = "✓" if diff < 0.01 else "✗"
    print(f"{'Hargreaves':<20} {hargreaves_orig:<15.4f} {hargreaves_new:<15.4f} {diff:<10.4f} {match}")
    if diff >= 0.01:
        all_match = False

    # Hamon (line 278) - uses calc_Es
    # calc_Es (line 663)
    def calc_svp(t):
        bracket = (17.27*t)/(t+237.3)
        return 0.6108 * np.exp(bracket)
    e_tmax = calc_svp(tmax_c)
    e_tmin = calc_svp(tmin_c)
    es = (e_tmax + e_tmin)/2
    bottom = tmean+273.3
    bracket = es/bottom
    hamon_orig = 29.8*daylength_orig * bracket
    hamon_new = pet_calc.hamon(tmax_c, tmin_c, daylength_new)
    diff = abs(hamon_orig - hamon_new)
    match = "✓" if diff < 0.01 else "✗"
    print(f"{'Hamon':<20} {hamon_orig:<15.4f} {hamon_new:<15.4f} {diff:<10.4f} {match}")
    if diff >= 0.01:
        all_match = False

    # Oudin (line 315)
    top = Ra_orig * (tmean + 5) * 0.408
    bottom = 100
    oudin_orig = top/bottom if tmean > -5 else 0
    oudin_new = pet_calc.oudin(tmax_c, tmin_c, Ra_new)
    diff = abs(oudin_orig - oudin_new)
    match = "✓" if diff < 0.01 else "✗"
    print(f"{'Oudin':<20} {oudin_orig:<15.4f} {oudin_new:<15.4f} {diff:<10.4f} {match}")
    if diff >= 0.01:
        all_match = False

    print("-"*70)

    if all_match:
        print("\n✓ All calculations match original script!")
        return True
    else:
        print("\n✗ Some calculations differ from original script")
        return False


def compare_snow_functions():
    """Compare snow model with original equations"""
    print("\n" + "="*70)
    print("Comparing Snow Model with Original Equations")
    print("="*70)

    # Create model
    snow_new = SnowModel(
        melt_thresh_temp=-6.0,
        melt_factor=0.35,
        precip_fraction=0.167
    )

    # Parameters
    melt_thresh_temperature = -6.0
    melt_factor = 0.35
    precip_fraction = 0.167

    print("\nTest parameters:")
    print(f"  Melt threshold: {melt_thresh_temperature}°C")
    print(f"  Melt factor: {melt_factor} mm/°C/day")
    print(f"  Precip fraction: {precip_fraction}")

    # Original functions from wb_ai_train.py (lines 903-926)
    def melt_one_day_orig(start_swe, tavg):
        """Line 908"""
        if tavg <= melt_thresh_temperature:
            return start_swe
        else:
            swe_delta = (tavg - melt_thresh_temperature) * melt_factor
            end_swe = start_swe - swe_delta
            if end_swe < 0:
                end_swe = 0
            return end_swe

    def accum_one_day_orig(start_swe, tavg, precip):
        """Line 917"""
        if tavg <= melt_thresh_temperature:
            rain = 0.0
        elif tavg > (melt_thresh_temperature + 6):
            rain = 1.0
        else:
            rain = precip_fraction * (tavg - melt_thresh_temperature)
        snow_increment = (1.0 - rain) * precip
        if snow_increment < 0:
            snow_increment = 0
        end_swe = start_swe + snow_increment
        return end_swe

    def est_snow_orig(swe, tavg, precip):
        """Line 903"""
        swe = melt_one_day_orig(swe, tavg)
        swe = accum_one_day_orig(swe, tavg, precip)
        return swe

    # Test scenarios
    test_cases = [
        ("Cold accumulation", 0.0, -10.0, 10.0),
        ("Cold no precip", 50.0, -8.0, 0.0),
        ("Warm melt", 50.0, 5.0, 0.0),
        ("Transition mixed", 20.0, 0.0, 10.0),
    ]

    print("\n" + "-"*70)
    print(f"{'Scenario':<20} {'Initial':<10} {'T(°C)':<8} {'P(mm)':<8} {'Original':<12} {'Refactored':<12} {'Match'}")
    print("-"*70)

    all_match = True
    for scenario, init_swe, tavg, precip in test_cases:
        swe_orig = est_snow_orig(init_swe, tavg, precip)
        swe_new = snow_new.estimate_snow(init_swe, tavg, precip)
        diff = abs(swe_orig - swe_new)
        match = "✓" if diff < 0.01 else "✗"
        print(f"{scenario:<20} {init_swe:<10.2f} {tavg:<8.1f} {precip:<8.2f} {swe_orig:<12.4f} {swe_new:<12.4f} {match}")
        if diff >= 0.01:
            all_match = False

    print("-"*70)

    if all_match:
        print("\n✓ Snow model matches original!")
        return True
    else:
        print("\n✗ Snow model differs from original")
        return False


def main():
    """Run all cross-validation tests"""
    print("\n" + "="*70)
    print("CROSS-VALIDATION AGAINST ORIGINAL SCRIPT")
    print("="*70)
    print("\nComparing water_balance.py against mike/wb_ai_train.py\n")

    tests = [
        ("PET Calculations", compare_pet_calculations),
        ("Snow Model", compare_snow_functions),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n✗ {test_name} ERROR: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "="*70)
    print("CROSS-VALIDATION SUMMARY")
    print("="*70)
    print(f"Tests passed: {passed}/{len(tests)}")
    print(f"Tests failed: {failed}/{len(tests)}")

    if failed == 0:
        print("\n✓ Refactored code produces identical results to original!")
        return 0
    else:
        print(f"\n✗ {failed} test(s) failed or showed differences")
        return 1


if __name__ == '__main__':
    sys.exit(main())
