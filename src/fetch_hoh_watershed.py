#!/usr/bin/env python3
"""
Fetch Hoh River watershed boundary using NLDI.
"""

import geopandas as gpd
from pynhd import NLDI
import matplotlib.pyplot as plt

def fetch_hoh_watershed(gage_id='12041200', output_dir='../data/watershed'):
    """
    Fetches the upstream basin boundary for Hoh River USGS gage.

    Args:
        gage_id (str): USGS gage ID for Hoh River near Forks, WA
        output_dir (str): Directory to save outputs
    """
    print(f"Querying NLDI for USGS Gage {gage_id} (Hoh River near Forks, WA)...")

    nldi = NLDI()

    try:
        basin = nldi.get_basins(feature_ids=gage_id, fsource="nwissite", split_catchment=False)

        if basin.empty:
            print("Error: No basin found. Check the Gage ID.")
            return None

        print(f"Basin retrieved! Area: {basin.geometry.area.iloc[0]:.4f} (degrees^2 approx)")

        # Save to file
        output_file = f"{output_dir}/hoh_watershed.geojson"
        basin.to_file(output_file, driver='GeoJSON')
        print(f"Saved watershed boundary to: {output_file}")

        # Get bounds for climate data downloading
        bounds = basin.total_bounds  # [minx, miny, maxx, maxy]
        print(f"\nWatershed bounds:")
        print(f"  Longitude: {bounds[0]:.4f} to {bounds[2]:.4f}")
        print(f"  Latitude:  {bounds[1]:.4f} to {bounds[3]:.4f}")

        # Quick Plot
        fig, ax = plt.subplots(figsize=(10, 10))
        basin.plot(ax=ax, color='lightblue', edgecolor='black', alpha=0.6, linewidth=2)
        plt.title(f"Hoh River Watershed\nUSGS {gage_id}", fontsize=14, fontweight='bold')
        plt.xlabel("Longitude", fontsize=12)
        plt.ylabel("Latitude", fontsize=12)
        plt.grid(True, alpha=0.3)

        # Save plot
        plot_file = f"{output_dir}/hoh_watershed_preview.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"Saved preview image: {plot_file}")
        plt.close()

        return basin

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Ensure you have 'pynhd' and 'geopandas' installed.")
        return None

if __name__ == "__main__":
    basin = fetch_hoh_watershed()

    if basin is not None:
        print("\n" + "="*80)
        print("SUCCESS! Hoh River watershed boundary fetched.")
        print("="*80)
        print("\nNext steps:")
        print("  1. Use bounds to download GridMET climate data")
        print("  2. Find SNOTEL sites within watershed")
        print("  3. Calculate elevation bands for climate stratification")
