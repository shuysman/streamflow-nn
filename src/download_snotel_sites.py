#!/usr/bin/env python3
"""
Download shapefile of all SNOTEL sites.

Data Source: NRCS SNOTEL Network
"""

import requests
import geopandas as gpd
import pandas as pd
from pathlib import Path
import json

OUTPUT_DIR = Path(__file__).parent.parent / 'data' / 'snotel'

def download_snotel_sites():
    """Download SNOTEL site locations from NRCS."""
    
    print("Downloading SNOTEL site locations...")
    print("=" * 80)
    
    # NRCS provides SNOTEL locations via their web service
    # We'll use the AWDB web service to get all sites
    url = "https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1/stations"
    
    params = {
        'stationTriplets': '*:*:SNTL',  # All SNOTEL stations
        'activeOnly': 'false',  # Include inactive sites
        'logicalAnd': 'true'
    }
    
    try:
        print(f"Fetching from: {url}")
        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()
        
        data = response.json()
        print(f"✓ Retrieved {len(data)} SNOTEL sites")
        
        # Parse into GeoDataFrame
        records = []
        for site in data:
            if 'latitude' in site and 'longitude' in site:
                records.append({
                    'station_id': site.get('stationId'),
                    'name': site.get('name'),
                    'state': site.get('stateCode'),
                    'county': site.get('countyName'),
                    'elevation_ft': site.get('elevation'),
                    'latitude': site.get('latitude'),
                    'longitude': site.get('longitude'),
                    'huc': site.get('huc'),
                    'actonId': site.get('actonId'),
                    'shefId': site.get('shefId'),
                    'beginDate': site.get('beginDate'),
                    'endDate': site.get('endDate'),
                    'network': 'SNTL'
                })
        
        # Create GeoDataFrame
        df = pd.DataFrame(records)
        gdf = gpd.GeoDataFrame(
            df, 
            geometry=gpd.points_from_xy(df.longitude, df.latitude),
            crs='EPSG:4326'
        )
        
        # Save as shapefile
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        shapefile_path = OUTPUT_DIR / 'snotel_sites.shp'
        gdf.to_file(shapefile_path)
        print(f"✓ Saved shapefile: {shapefile_path}")
        
        # Also save as GeoJSON (easier to read)
        geojson_path = OUTPUT_DIR / 'snotel_sites.geojson'
        gdf.to_file(geojson_path, driver='GeoJSON')
        print(f"✓ Saved GeoJSON: {geojson_path}")
        
        # Also save as CSV (no geometry)
        csv_path = OUTPUT_DIR / 'snotel_sites.csv'
        df.to_csv(csv_path, index=False)
        print(f"✓ Saved CSV: {csv_path}")
        
        # Summary statistics
        print(f"\n" + "=" * 80)
        print("Summary:")
        print(f"  Total sites: {len(gdf)}")
        print(f"  States: {gdf['state'].nunique()}")
        print(f"  Elevation range: {gdf['elevation_ft'].min():.0f} - {gdf['elevation_ft'].max():.0f} ft")
        
        print(f"\nSites by state:")
        print(gdf['state'].value_counts().head(10))
        
        return gdf
        
    except Exception as e:
        print(f"✗ Error: {e}")
        print("\nTrying alternative method...")
        return download_snotel_sites_csv()


def download_snotel_sites_csv():
    """Alternative: Download from NRCS CSV export."""
    
    # NRCS provides a CSV inventory
    url = "https://wcc.sc.egov.usda.gov/nwcc/inventory"
    
    print(f"Downloading from: {url}")
    
    try:
        # This downloads the full inventory page
        # We need to parse it to extract SNOTEL sites
        
        # For now, use a pre-compiled list
        print("Using NRCS AWDB REST API for individual state queries...")
        
        states = ['WY', 'MT', 'ID', 'CO', 'UT', 'NM', 'AZ', 'NV', 'CA', 'OR', 'WA', 'AK']
        all_sites = []
        
        for state in states:
            state_url = f"https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1/stations"
            params = {
                'stationTriplets': f'*:{state}:SNTL',
                'activeOnly': 'false'
            }
            
            try:
                response = requests.get(state_url, params=params, timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    all_sites.extend(data)
                    print(f"  {state}: {len(data)} sites")
            except:
                pass
        
        print(f"\n✓ Retrieved {len(all_sites)} total SNOTEL sites")
        
        # Parse into GeoDataFrame
        records = []
        for site in all_sites:
            if 'latitude' in site and 'longitude' in site:
                records.append({
                    'station_id': site.get('stationId'),
                    'name': site.get('name'),
                    'state': site.get('stateCode'),
                    'county': site.get('countyName'),
                    'elevation_ft': site.get('elevation'),
                    'latitude': site.get('latitude'),
                    'longitude': site.get('longitude'),
                    'huc': site.get('huc'),
                    'network': 'SNTL'
                })
        
        df = pd.DataFrame(records)
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df.longitude, df.latitude),
            crs='EPSG:4326'
        )
        
        # Save files
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        shapefile_path = OUTPUT_DIR / 'snotel_sites.shp'
        gdf.to_file(shapefile_path)
        
        geojson_path = OUTPUT_DIR / 'snotel_sites.geojson'
        gdf.to_file(geojson_path, driver='GeoJSON')
        
        csv_path = OUTPUT_DIR / 'snotel_sites.csv'
        df.to_csv(csv_path, index=False)
        
        print(f"\n✓ Saved all formats")
        print(f"  Shapefile: {shapefile_path}")
        print(f"  GeoJSON: {geojson_path}")
        print(f"  CSV: {csv_path}")
        
        return gdf
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return None


if __name__ == '__main__':
    gdf = download_snotel_sites()
    
    if gdf is not None:
        print(f"\n" + "=" * 80)
        print("✓ SNOTEL sites shapefile download complete!")
        print("=" * 80)
