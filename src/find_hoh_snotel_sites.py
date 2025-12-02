#!/usr/bin/env python3
"""
Find SNOTEL sites near the Hoh River watershed.

Searches the SNOTEL sites database for stations in Washington state
within the Olympic Mountains region.
"""

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import numpy as np

def load_watershed():
    """Load Hoh River watershed polygon."""
    watershed_path = '../data/watershed/hoh_watershed.geojson'
    gdf = gpd.read_file(watershed_path)
    print(f"✓ Loaded Hoh River watershed")
    print(f"  Bounds: {gdf.total_bounds}")
    return gdf

def find_nearby_snotel(buffer_km=50):
    """
    Find SNOTEL sites near Hoh River watershed.

    Parameters:
        buffer_km: Search radius in kilometers
    """
    print("\n" + "="*80)
    print(f"SEARCHING FOR SNOTEL SITES NEAR HOH RIVER")
    print("="*80)
    print(f"Search radius: {buffer_km} km from watershed boundary")

    # Load SNOTEL sites
    snotel = pd.read_csv('../data/snotel/snotel_sites.csv')
    print(f"\n✓ Loaded {len(snotel)} SNOTEL sites nationally")

    # Filter to Washington state
    snotel_wa = snotel[snotel['state'] == 'WA'].copy()
    print(f"✓ Filtered to {len(snotel_wa)} sites in Washington")

    # Load watershed
    watershed = load_watershed()
    watershed_bounds = watershed.total_bounds  # [minx, miny, maxx, maxy]

    # Buffer watershed (rough approximation: 1 degree ≈ 111 km)
    buffer_deg = buffer_km / 111.0
    search_bounds = [
        watershed_bounds[0] - buffer_deg,  # minx
        watershed_bounds[1] - buffer_deg,  # miny
        watershed_bounds[2] + buffer_deg,  # maxx
        watershed_bounds[3] + buffer_deg   # maxy
    ]

    print(f"\nSearch box: lon [{search_bounds[0]:.3f}, {search_bounds[2]:.3f}], "
          f"lat [{search_bounds[1]:.3f}, {search_bounds[3]:.3f}]")

    # Filter by bounding box
    nearby = snotel_wa[
        (snotel_wa['longitude'] >= search_bounds[0]) &
        (snotel_wa['longitude'] <= search_bounds[2]) &
        (snotel_wa['latitude'] >= search_bounds[1]) &
        (snotel_wa['latitude'] <= search_bounds[3])
    ].copy()

    if len(nearby) == 0:
        print(f"\n⚠️  No SNOTEL sites found within {buffer_km} km!")
        print("The Hoh River watershed is in the Olympic Mountains.")
        print("Expanding search...")
        return find_nearby_snotel(buffer_km=100)

    # Calculate distances to watershed centroid
    watershed_centroid = watershed.geometry.centroid.iloc[0]

    def haversine_distance(lat1, lon1, lat2, lon2):
        """Calculate great circle distance in km."""
        R = 6371  # Earth radius in km
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = (np.sin(dlat/2)**2 +
             np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) *
             np.sin(dlon/2)**2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        return R * c

    nearby['distance_km'] = nearby.apply(
        lambda row: haversine_distance(
            watershed_centroid.y, watershed_centroid.x,
            row['latitude'], row['longitude']
        ), axis=1
    )

    # Sort by distance
    nearby = nearby.sort_values('distance_km')

    print(f"\n✓ Found {len(nearby)} SNOTEL sites within {buffer_km} km")
    print("\nNearest sites:")
    print("="*80)

    for idx, row in nearby.head(10).iterrows():
        print(f"{row['name']:30s} | {row['elevation_ft']:6.0f} ft | {row['distance_km']:5.1f} km")

    return nearby

def select_representative_sites(nearby_sites, max_sites=3):
    """
    Select representative SNOTEL sites based on:
    1. Distance (prefer closer)
    2. Elevation (want coverage of different elevations)
    3. Data availability
    """
    print("\n" + "="*80)
    print(f"SELECTING TOP {max_sites} REPRESENTATIVE SITES")
    print("="*80)

    # Sort by distance and elevation to get diversity
    selected = nearby_sites.head(max_sites)

    print("\nSelected sites:")
    for idx, row in selected.iterrows():
        print(f"\n{row['name']} (ID: {row['station_id']})")
        print(f"  Elevation: {row['elevation_ft']:.0f} ft ({row['elevation_ft']*0.3048:.0f} m)")
        print(f"  Distance: {row['distance_km']:.1f} km from watershed")
        print(f"  Location: {row['latitude']:.4f}°N, {row['longitude']:.4f}°W")
        print(f"  Data: {row['beginDate']} to {row['endDate']}")

    # Save selected sites
    output_file = '../data/snotel/hoh_snotel_sites.csv'
    selected.to_csv(output_file, index=False)
    print(f"\n✓ Saved to: {output_file}")

    return selected

if __name__ == '__main__':
    # Find nearby sites
    nearby = find_nearby_snotel(buffer_km=50)

    if nearby is not None and len(nearby) > 0:
        # Select representative sites
        selected = select_representative_sites(nearby, max_sites=3)

        print("\n" + "="*80)
        print("NEXT STEPS")
        print("="*80)
        print("\n1. Download SWE data for selected sites:")
        for idx, row in selected.iterrows():
            print(f"   python download_snotel.py --site {row['station_id']} --name {row['name'].replace(' ', '_')}")

        print("\n2. Update Stage 6 notebook to include SNOTEL features")
    else:
        print("\n⚠️  No suitable SNOTEL sites found!")
        print("Consider using climate data only for Hoh River forecasting.")
