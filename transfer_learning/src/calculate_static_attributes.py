#!/usr/bin/env python3
"""
Calculate static watershed attributes for NeuralHydrology transfer learning.

This script determines the 7 CAMELS-style static attributes needed for
fine-tuning pre-trained models on new watersheds:
  - elev_mean: Mean elevation (m)
  - slope_mean: Mean slope (degrees)
  - area_gages2: Drainage area (km²)
  - frac_forest: Forest fraction (0-1)
  - soil_depth_pelletier: Soil depth (m)
  - sand_frac: Sand fraction (0-1)
  - clay_frac: Clay fraction (0-1)

Data sources:
  - USGS NWIS: Drainage area, site info
  - Open-Elevation API: Elevation sampling
  - SoilGrids API: Soil properties
  - Watershed boundary: Area calculation from GeoJSON

Usage:
    python calculate_static_attributes.py --watershed hoh --usgs-site 12041200
    python calculate_static_attributes.py --geojson data/watershed/hoh_watershed.geojson
    python calculate_static_attributes.py --interactive
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests


def get_usgs_site_info(site_no: str) -> dict:
    """Fetch site information from USGS NWIS."""
    url = "https://waterservices.usgs.gov/nwis/site/"
    params = {
        "format": "rdb",
        "sites": site_no,
        "siteOutput": "expanded",
        "siteStatus": "all"
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()

        # Parse RDB format (tab-separated with comment headers)
        lines = response.text.strip().split('\n')
        data_lines = [l for l in lines if not l.startswith('#')]

        if len(data_lines) >= 3:
            headers = data_lines[0].split('\t')
            values = data_lines[2].split('\t')  # Skip format line
            site_data = dict(zip(headers, values))

            return {
                'site_no': site_data.get('site_no', ''),
                'station_nm': site_data.get('station_nm', ''),
                'lat': float(site_data.get('dec_lat_va', 0)),
                'lon': float(site_data.get('dec_long_va', 0)),
                'drainage_area_sqmi': float(site_data.get('drain_area_va', 0)) if site_data.get('drain_area_va') else None,
                'altitude': float(site_data.get('alt_va', 0)) if site_data.get('alt_va') else None,
            }
    except Exception as e:
        print(f"Warning: Could not fetch USGS site info: {e}")

    return {}


def calculate_polygon_area_km2(coordinates: list) -> float:
    """
    Calculate polygon area in km² using the Shoelace formula.
    Assumes coordinates are in [lon, lat] format (WGS84).
    Uses an approximate conversion for mid-latitudes.
    """
    # Extract the outer ring (first element of coordinates)
    ring = coordinates[0] if coordinates and isinstance(coordinates[0][0], list) else coordinates

    # Convert to numpy arrays
    lons = np.array([p[0] for p in ring])
    lats = np.array([p[1] for p in ring])

    # Approximate conversion to meters at the centroid latitude
    mean_lat = np.mean(lats)
    lat_to_m = 111132.92  # meters per degree latitude
    lon_to_m = 111132.92 * np.cos(np.radians(mean_lat))  # varies with latitude

    # Convert to local meters
    x = (lons - lons[0]) * lon_to_m
    y = (lats - lats[0]) * lat_to_m

    # Shoelace formula
    n = len(x)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += x[i] * y[j]
        area -= x[j] * y[i]
    area = abs(area) / 2.0

    # Convert m² to km²
    return area / 1e6


def load_watershed_boundary(geojson_path: str) -> dict:
    """Load watershed boundary from GeoJSON file."""
    with open(geojson_path, 'r') as f:
        data = json.load(f)

    # Extract polygon coordinates
    if data['type'] == 'FeatureCollection':
        feature = data['features'][0]
    else:
        feature = data

    geometry = feature['geometry']
    coordinates = geometry['coordinates']

    # Calculate centroid for sampling
    ring = coordinates[0] if coordinates and isinstance(coordinates[0][0], list) else coordinates
    lons = [p[0] for p in ring]
    lats = [p[1] for p in ring]

    return {
        'coordinates': coordinates,
        'centroid_lon': np.mean(lons),
        'centroid_lat': np.mean(lats),
        'bounds': {
            'min_lon': min(lons),
            'max_lon': max(lons),
            'min_lat': min(lats),
            'max_lat': max(lats)
        },
        'area_km2': calculate_polygon_area_km2(coordinates)
    }


def sample_elevations_open_elevation(lats: list, lons: list) -> list:
    """
    Query Open-Elevation API for elevation at multiple points.
    Free, no API key required, but rate limited.
    """
    url = "https://api.open-elevation.com/api/v1/lookup"

    # Batch requests (API accepts up to 100 points)
    locations = [{"latitude": lat, "longitude": lon} for lat, lon in zip(lats, lons)]

    try:
        response = requests.post(url, json={"locations": locations}, timeout=60)
        response.raise_for_status()
        results = response.json()['results']
        return [r['elevation'] for r in results]
    except Exception as e:
        print(f"Warning: Open-Elevation API failed: {e}")
        return []


def sample_elevations_openmeteo(lats: list, lons: list) -> list:
    """
    Query Open-Meteo elevation API.
    Very fast and reliable.
    """
    elevations = []

    # Open-Meteo accepts comma-separated coordinates
    lat_str = ",".join([f"{lat:.4f}" for lat in lats])
    lon_str = ",".join([f"{lon:.4f}" for lon in lons])

    url = f"https://api.open-meteo.com/v1/elevation?latitude={lat_str}&longitude={lon_str}"

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        elevations = data.get('elevation', [])
        if isinstance(elevations, (int, float)):
            elevations = [elevations]
    except Exception as e:
        print(f"Warning: Open-Meteo elevation API failed: {e}")

    return elevations


def calculate_elevation_stats(boundary: dict, n_samples: int = 100) -> dict:
    """
    Calculate elevation statistics by sampling points within the watershed.
    """
    bounds = boundary['bounds']

    # Generate random sample points within bounding box
    np.random.seed(42)  # Reproducibility
    sample_lats = np.random.uniform(bounds['min_lat'], bounds['max_lat'], n_samples * 2)
    sample_lons = np.random.uniform(bounds['min_lon'], bounds['max_lon'], n_samples * 2)

    # For simplicity, use all points (ideally would do point-in-polygon test)
    # Take first n_samples points
    lats = list(sample_lats[:n_samples])
    lons = list(sample_lons[:n_samples])

    # Try Open-Meteo first (faster, more reliable)
    print(f"  Sampling {n_samples} elevation points via Open-Meteo...")
    elevations = sample_elevations_openmeteo(lats, lons)

    if not elevations:
        print("  Falling back to Open-Elevation API...")
        elevations = sample_elevations_open_elevation(lats, lons)

    if not elevations:
        print("  Warning: Could not fetch elevations, using estimate from centroid")
        # Last resort: single point elevation
        single_elev = sample_elevations_openmeteo(
            [boundary['centroid_lat']],
            [boundary['centroid_lon']]
        )
        if single_elev:
            return {'elev_mean': single_elev[0], 'elev_std': 0}
        return {'elev_mean': None, 'elev_std': None}

    # Filter out invalid elevations (e.g., ocean = -9999)
    valid_elevations = [e for e in elevations if e > -1000]

    if valid_elevations:
        return {
            'elev_mean': np.mean(valid_elevations),
            'elev_std': np.std(valid_elevations),
            'elev_min': np.min(valid_elevations),
            'elev_max': np.max(valid_elevations),
            'n_samples': len(valid_elevations)
        }

    return {'elev_mean': None, 'elev_std': None}


def estimate_slope_from_elevation(elev_stats: dict, area_km2: float) -> float:
    """
    Estimate mean slope from elevation range and area.
    This is a rough approximation based on typical watershed geometry.

    slope ≈ arctan(elevation_range / characteristic_length)
    where characteristic_length ≈ sqrt(area)
    """
    if not elev_stats.get('elev_max') or not elev_stats.get('elev_min'):
        return None

    elev_range = elev_stats['elev_max'] - elev_stats['elev_min']

    # Characteristic length in meters
    char_length_m = np.sqrt(area_km2) * 1000

    # Estimate slope (this tends to underestimate actual mean slope)
    # Apply a scaling factor based on typical mountain watersheds
    raw_slope = np.degrees(np.arctan(elev_range / char_length_m))

    # Scale up - actual mean slopes are typically 2-4x steeper than this estimate
    estimated_slope = raw_slope * 3.0

    return min(estimated_slope, 45.0)  # Cap at 45 degrees


def get_soilgrids_data(lat: float, lon: float) -> dict:
    """
    Query SoilGrids API for soil properties.
    Returns sand/clay fractions and soil depth estimates.
    """
    # SoilGrids REST API
    url = f"https://rest.isric.org/soilgrids/v2.0/properties/query"
    params = {
        "lon": lon,
        "lat": lat,
        "property": ["sand", "clay", "bdod"],  # sand, clay, bulk density
        "depth": ["0-5cm", "5-15cm", "15-30cm", "30-60cm"],
        "value": "mean"
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        # Extract sand and clay percentages (average across depths)
        sand_values = []
        clay_values = []

        for layer in data.get('properties', {}).get('layers', []):
            name = layer.get('name', '')
            for depth in layer.get('depths', []):
                value = depth.get('values', {}).get('mean')
                if value is not None:
                    if name == 'sand':
                        sand_values.append(value / 10.0)  # Convert g/kg to %
                    elif name == 'clay':
                        clay_values.append(value / 10.0)

        sand_frac = np.mean(sand_values) / 100 if sand_values else None
        clay_frac = np.mean(clay_values) / 100 if clay_values else None

        return {
            'sand_frac': sand_frac,
            'clay_frac': clay_frac,
        }

    except Exception as e:
        print(f"Warning: SoilGrids API failed: {e}")
        return {}


def estimate_forest_fraction(lat: float, lon: float, elev_mean: float = None) -> float:
    """
    Estimate forest fraction based on location and elevation.
    This is a rough heuristic - ideally use NLCD or similar land cover data.
    """
    # Pacific Northwest (high rainfall, high forest cover)
    if -125 < lon < -120 and 45 < lat < 50:
        base_forest = 0.85
    # Rocky Mountains
    elif -115 < lon < -105 and 40 < lat < 50:
        base_forest = 0.55
    # General US temperate
    elif -125 < lon < -70 and 30 < lat < 50:
        base_forest = 0.45
    else:
        base_forest = 0.40

    # Adjust for elevation (less forest at high elevations)
    if elev_mean:
        if elev_mean > 3000:
            base_forest *= 0.3  # Alpine, minimal forest
        elif elev_mean > 2500:
            base_forest *= 0.6  # Subalpine
        elif elev_mean > 2000:
            base_forest *= 0.8  # Upper montane

    return min(base_forest, 1.0)


def estimate_soil_depth(lat: float, lon: float, slope_mean: float = None) -> float:
    """
    Estimate soil depth based on location and slope.
    Based on Pelletier et al. (2016) patterns.
    """
    # Base soil depth (meters)
    base_depth = 1.5

    # Steeper slopes = thinner soils
    if slope_mean:
        if slope_mean > 30:
            base_depth *= 0.5
        elif slope_mean > 20:
            base_depth *= 0.7
        elif slope_mean > 10:
            base_depth *= 0.85

    # Regional adjustments
    # Pacific Northwest - high weathering, moderate depths
    if -125 < lon < -120 and 45 < lat < 50:
        base_depth *= 1.1
    # Rocky Mountains - thin soils
    elif -115 < lon < -105:
        base_depth *= 0.8

    return max(base_depth, 0.3)  # Minimum 30cm


def calculate_all_attributes(
    watershed_name: str,
    geojson_path: str = None,
    usgs_site: str = None,
    manual_overrides: dict = None
) -> dict:
    """
    Calculate all static attributes for a watershed.

    Args:
        watershed_name: Name for the basin (e.g., 'hoh_lumped')
        geojson_path: Path to watershed boundary GeoJSON
        usgs_site: USGS site number for drainage area lookup
        manual_overrides: Dict of manually specified values

    Returns:
        Dict with all 7 CAMELS attributes
    """
    manual_overrides = manual_overrides or {}
    attributes = {'basin': watershed_name}

    # Initialize with None
    lat, lon = None, None
    area_km2 = None
    elev_stats = {}

    # 1. Load watershed boundary if provided
    if geojson_path and Path(geojson_path).exists():
        print(f"Loading watershed boundary from {geojson_path}...")
        boundary = load_watershed_boundary(geojson_path)
        lat = boundary['centroid_lat']
        lon = boundary['centroid_lon']
        area_km2 = boundary['area_km2']
        print(f"  Centroid: ({lat:.4f}, {lon:.4f})")
        print(f"  Area from polygon: {area_km2:.1f} km²")

        # Calculate elevation stats
        print("Calculating elevation statistics...")
        elev_stats = calculate_elevation_stats(boundary, n_samples=50)
        if elev_stats.get('elev_mean'):
            print(f"  Mean elevation: {elev_stats['elev_mean']:.0f} m")
            print(f"  Elevation range: {elev_stats.get('elev_min', 0):.0f} - {elev_stats.get('elev_max', 0):.0f} m")

    # 2. Get USGS site info if provided
    if usgs_site:
        print(f"Fetching USGS site info for {usgs_site}...")
        site_info = get_usgs_site_info(usgs_site)
        if site_info:
            print(f"  Station: {site_info.get('station_nm', 'Unknown')}")
            if site_info.get('drainage_area_sqmi'):
                usgs_area_km2 = site_info['drainage_area_sqmi'] * 2.58999  # sq mi to km²
                print(f"  USGS drainage area: {usgs_area_km2:.1f} km²")
                if area_km2:
                    print(f"  (Polygon area was {area_km2:.1f} km² - using USGS value)")
                area_km2 = usgs_area_km2
            if not lat and site_info.get('lat'):
                lat = site_info['lat']
                lon = site_info['lon']

    # 3. Set drainage area
    if 'area_gages2' in manual_overrides:
        attributes['area_gages2'] = manual_overrides['area_gages2']
    elif area_km2:
        attributes['area_gages2'] = round(area_km2, 1)
    else:
        attributes['area_gages2'] = None

    # 4. Set elevation
    if 'elev_mean' in manual_overrides:
        attributes['elev_mean'] = manual_overrides['elev_mean']
    elif elev_stats.get('elev_mean'):
        attributes['elev_mean'] = round(elev_stats['elev_mean'], 1)
    else:
        attributes['elev_mean'] = None

    # 5. Calculate/estimate slope
    if 'slope_mean' in manual_overrides:
        attributes['slope_mean'] = manual_overrides['slope_mean']
    else:
        slope = estimate_slope_from_elevation(elev_stats, area_km2 or 100)
        attributes['slope_mean'] = round(slope, 1) if slope else None
        if slope:
            print(f"  Estimated slope: {slope:.1f}°")

    # 6. Get soil properties
    if lat and lon:
        print("Fetching soil properties from SoilGrids...")
        soil_data = get_soilgrids_data(lat, lon)
        time.sleep(0.5)  # Rate limiting

        if 'sand_frac' in manual_overrides:
            attributes['sand_frac'] = manual_overrides['sand_frac']
        elif soil_data.get('sand_frac'):
            attributes['sand_frac'] = round(soil_data['sand_frac'], 2)
            print(f"  Sand fraction: {attributes['sand_frac']:.2f}")
        else:
            attributes['sand_frac'] = None

        if 'clay_frac' in manual_overrides:
            attributes['clay_frac'] = manual_overrides['clay_frac']
        elif soil_data.get('clay_frac'):
            attributes['clay_frac'] = round(soil_data['clay_frac'], 2)
            print(f"  Clay fraction: {attributes['clay_frac']:.2f}")
        else:
            attributes['clay_frac'] = None

    # 7. Estimate soil depth
    if 'soil_depth_pelletier' in manual_overrides:
        attributes['soil_depth_pelletier'] = manual_overrides['soil_depth_pelletier']
    else:
        soil_depth = estimate_soil_depth(lat or 45, lon or -120, attributes.get('slope_mean'))
        attributes['soil_depth_pelletier'] = round(soil_depth, 2)
        print(f"  Estimated soil depth: {soil_depth:.2f} m")

    # 8. Estimate forest fraction
    if 'frac_forest' in manual_overrides:
        attributes['frac_forest'] = manual_overrides['frac_forest']
    else:
        frac_forest = estimate_forest_fraction(lat or 45, lon or -120, attributes.get('elev_mean'))
        attributes['frac_forest'] = round(frac_forest, 2)
        print(f"  Estimated forest fraction: {frac_forest:.2f}")

    return attributes


def save_attributes_csv(attributes: dict, output_path: str):
    """Save attributes to CSV in NeuralHydrology format."""
    columns = ['basin', 'elev_mean', 'slope_mean', 'area_gages2',
               'frac_forest', 'soil_depth_pelletier', 'sand_frac', 'clay_frac']

    df = pd.DataFrame([attributes])
    df = df[columns]  # Ensure column order
    df.to_csv(output_path, index=False)
    print(f"\nSaved attributes to {output_path}")


def interactive_mode():
    """Run in interactive mode with user prompts."""
    print("=" * 60)
    print("Static Watershed Attributes Calculator")
    print("=" * 60)

    watershed_name = input("\nEnter watershed name (e.g., 'hoh_lumped'): ").strip()

    geojson_path = input("Enter path to watershed GeoJSON (or press Enter to skip): ").strip()
    if not geojson_path:
        geojson_path = None

    usgs_site = input("Enter USGS site number (or press Enter to skip): ").strip()
    if not usgs_site:
        usgs_site = None

    # Calculate attributes
    attributes = calculate_all_attributes(
        watershed_name=watershed_name,
        geojson_path=geojson_path,
        usgs_site=usgs_site
    )

    # Allow manual overrides
    print("\n" + "-" * 40)
    print("Review and override calculated values:")
    print("-" * 40)

    for key in ['elev_mean', 'slope_mean', 'area_gages2', 'frac_forest',
                'soil_depth_pelletier', 'sand_frac', 'clay_frac']:
        current = attributes.get(key)
        override = input(f"  {key} [{current}]: ").strip()
        if override:
            try:
                attributes[key] = float(override)
            except ValueError:
                print(f"    Invalid value, keeping {current}")

    # Save
    output_path = input("\nOutput CSV path [transfer_learning/data/neuralhydrology/attributes/{name}_attributes.csv]: ").strip()
    if not output_path:
        output_path = f"transfer_learning/data/neuralhydrology/attributes/{watershed_name.split('_')[0]}_attributes.csv"

    save_attributes_csv(attributes, output_path)

    return attributes


def main():
    parser = argparse.ArgumentParser(
        description="Calculate static watershed attributes for NeuralHydrology"
    )
    parser.add_argument(
        '--watershed', '-w',
        help="Watershed name (e.g., 'hoh', 'lamar')"
    )
    parser.add_argument(
        '--geojson', '-g',
        help="Path to watershed boundary GeoJSON file"
    )
    parser.add_argument(
        '--usgs-site', '-u',
        help="USGS site number for drainage area lookup"
    )
    parser.add_argument(
        '--output', '-o',
        help="Output CSV path"
    )
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help="Run in interactive mode"
    )

    # Manual override arguments
    parser.add_argument('--elev-mean', type=float, help="Manual elevation (m)")
    parser.add_argument('--slope-mean', type=float, help="Manual slope (degrees)")
    parser.add_argument('--area', type=float, help="Manual drainage area (km²)")
    parser.add_argument('--frac-forest', type=float, help="Manual forest fraction (0-1)")
    parser.add_argument('--soil-depth', type=float, help="Manual soil depth (m)")
    parser.add_argument('--sand-frac', type=float, help="Manual sand fraction (0-1)")
    parser.add_argument('--clay-frac', type=float, help="Manual clay fraction (0-1)")

    args = parser.parse_args()

    if args.interactive:
        interactive_mode()
        return

    if not args.watershed:
        parser.print_help()
        print("\nError: --watershed is required (or use --interactive)")
        sys.exit(1)

    # Build manual overrides
    overrides = {}
    if args.elev_mean is not None:
        overrides['elev_mean'] = args.elev_mean
    if args.slope_mean is not None:
        overrides['slope_mean'] = args.slope_mean
    if args.area is not None:
        overrides['area_gages2'] = args.area
    if args.frac_forest is not None:
        overrides['frac_forest'] = args.frac_forest
    if args.soil_depth is not None:
        overrides['soil_depth_pelletier'] = args.soil_depth
    if args.sand_frac is not None:
        overrides['sand_frac'] = args.sand_frac
    if args.clay_frac is not None:
        overrides['clay_frac'] = args.clay_frac

    # Default geojson path
    geojson_path = args.geojson
    if not geojson_path:
        # Try standard locations
        for path in [
            f"data/watershed/{args.watershed}_watershed.geojson",
            f"transfer_learning/data/watershed/{args.watershed}_watershed.geojson"
        ]:
            if Path(path).exists():
                geojson_path = path
                break

    # Calculate
    watershed_name = f"{args.watershed}_lumped"
    print(f"\nCalculating attributes for {watershed_name}")
    print("=" * 50)

    attributes = calculate_all_attributes(
        watershed_name=watershed_name,
        geojson_path=geojson_path,
        usgs_site=args.usgs_site,
        manual_overrides=overrides
    )

    # Print summary
    print("\n" + "=" * 50)
    print("CALCULATED ATTRIBUTES:")
    print("=" * 50)
    for key, value in attributes.items():
        if key != 'basin':
            print(f"  {key}: {value}")

    # Save
    output_path = args.output
    if not output_path:
        output_path = f"transfer_learning/data/neuralhydrology/attributes/{args.watershed}_attributes.csv"

    save_attributes_csv(attributes, output_path)


if __name__ == "__main__":
    main()
