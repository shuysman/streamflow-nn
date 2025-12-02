import geopandas as gpd
from pynhd import NLDI
import matplotlib.pyplot as plt

def fetch_lamar_watershed(gage_id, output_file='lamar_watershed.geojson'):
    """
    Fetches the upstream basin boundary for a USGS gage using the NLDI.
    
    Args:
        gage_id (str): The USGS gage ID (e.g., '06191500')
        output_file (str): Path to save the GeoJSON
    """
    print(f"Querying NLDI for USGS Gage {gage_id}...")
    
    # The NLDI allows us to query by 'feature source'. 
    # 'nwissite' is the source for USGS gages.
    nldi = NLDI()
    
    # We request the 'basin' associated with this feature
    # split_catchment=False gives the full upstream accumulated area
    try:
        basin = nldi.get_basins(feature_ids=gage_id, fsource="nwissite", split_catchment=False)
        
        if basin.empty:
            print("Error: No basin found. Check the Gage ID.")
            return

        print(f"Basin retrieved! Area: {basin.geometry.area.iloc[0]:.4f} (degrees^2 approx)")
        
        # Save to file
        basin.to_file(output_file, driver='GeoJSON')
        print(f"Saved watershed boundary to: {output_file}")
        
        # Quick Plot to Verify
        fig, ax = plt.subplots(figsize=(8, 8))
        basin.plot(ax=ax, color='lightblue', edgecolor='black', alpha=0.6)
        plt.title(f"Upstream Watershed: USGS {gage_id}")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.grid(True)
        
        # Save plot for verification
        plt.savefig("watershed_preview.png")
        print("Saved preview image: watershed_preview.png")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Ensure you have 'pynhd' and 'geopandas' installed.")
        print("pip install pynhd geopandas shapely matplotlib")

if __name__ == "__main__":
    # Lamar River near Tower Falls
    GAGE_ID = "06191500" 
    fetch_lamar_watershed(GAGE_ID)
