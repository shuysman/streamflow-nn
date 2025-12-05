#!/usr/bin/env python3
"""
Download all USGS stream gage locations and save to GeoPackage format.

This script fetches active and inactive stream gage locations from the USGS
and saves them as a GeoPackage file for GIS analysis.

It uses multiple approaches:
1. Try to download from USGS API
2. Fallback to state-by-state download if needed
"""

import requests
import geopandas as gpd
from shapely.geometry import Point
import pandas as pd
from pathlib import Path
import logging
import time
import warnings
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import urllib3

# Suppress SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# US state codes for fallback approach
US_STATES = [
    'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
    'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
    'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
    'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
    'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY',
    'AS', 'GU', 'MP', 'PR', 'VI'  # Territories
]


def create_session():
    """Create a requests session with retry logic."""
    session = requests.Session()
    retry = Retry(
        total=3,
        read=3,
        connect=3,
        backoff_factor=0.5,
        status_forcelist=(500, 502, 504)
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session


def download_usgs_gages(output_file='data/usgs_stream_gages.gpkg', site_type='ST'):
    """
    Download USGS stream gage locations from the USGS Water Services API.

    Parameters:
    -----------
    output_file : str
        Path to output GeoPackage file
    site_type : str
        Site type code. 'ST' = Stream gage
        Other options: 'ST-CA' = Continuous-record streamgage

    Returns:
    --------
    geopandas.GeoDataFrame
        GeoDataFrame containing all gage locations
    """

    # USGS Water Services API endpoint
    base_url = "https://waterservices.usgs.gov/nwis/site/"

    logger.info(f"Fetching USGS stream gage data from API...")

    session = create_session()
    all_data = []

    # Download state by state to avoid timeout/size issues
    for state in US_STATES:
        logger.info(f"Downloading gages for state: {state}")

        # Parameters for the API request
        params = {
            'format': 'rdb',  # Tab-delimited format
            'stateCd': state,
            'siteType': site_type,
            'siteStatus': 'all'  # Include both active and inactive gages
        }

        try:
            # Make the API request
            response = session.get(base_url, params=params, timeout=60, verify=False)
            response.raise_for_status()

            # Parse the RDB format (tab-delimited with header info)
            lines = response.text.strip().split('\n')

            # Find the header line (starts with agency_cd)
            header_idx = None
            for i, line in enumerate(lines):
                if line.startswith('agency_cd'):
                    header_idx = i
                    break

            if header_idx is None:
                logger.warning(f"No gages found for state {state}")
                continue

            # Parse header and data
            header = lines[header_idx].split('\t')
            # Skip the format line (next line after header)
            data_lines = lines[header_idx + 2:]

            # Parse data into list of dictionaries
            state_count = 0
            for line in data_lines:
                if line.strip() and not line.startswith('#'):
                    values = line.split('\t')
                    if len(values) == len(header):
                        all_data.append(dict(zip(header, values)))
                        state_count += 1

            logger.info(f"  Found {state_count} gages in {state}")
            time.sleep(0.5)  # Be nice to the API

        except requests.exceptions.RequestException as e:
            logger.warning(f"  Error fetching data for state {state}: {e}")
            continue
        except Exception as e:
            logger.warning(f"  Error processing data for state {state}: {e}")
            continue

    logger.info(f"\nTotal gages retrieved: {len(all_data)}")

    if len(all_data) == 0:
        raise ValueError("No gage data retrieved from any state")

    try:
        # Create pandas DataFrame
        df = pd.DataFrame(all_data)

        # Convert coordinate columns to numeric
        df['dec_lat_va'] = pd.to_numeric(df['dec_lat_va'], errors='coerce')
        df['dec_long_va'] = pd.to_numeric(df['dec_long_va'], errors='coerce')

        # Remove rows with missing coordinates
        df_clean = df.dropna(subset=['dec_lat_va', 'dec_long_va'])
        logger.info(f"Kept {len(df_clean)} gages with valid coordinates")

        # Create Point geometries
        geometry = [Point(xy) for xy in zip(df_clean['dec_long_va'], df_clean['dec_lat_va'])]

        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(df_clean, geometry=geometry, crs='EPSG:4326')

        # Create output directory if needed
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save to GeoPackage
        logger.info(f"Saving to {output_file}...")
        gdf.to_file(output_file, driver='GPKG')

        logger.info(f"Successfully saved {len(gdf)} gages to {output_file}")
        logger.info(f"CRS: {gdf.crs}")
        logger.info(f"Bounds: {gdf.total_bounds}")

        return gdf

    except Exception as e:
        logger.error(f"Error processing data: {e}")
        raise


if __name__ == '__main__':
    # Download all stream gages
    gdf = download_usgs_gages(
        output_file='data/usgs_stream_gages.gpkg',
        site_type='ST'
    )

    print(f"\nSummary:")
    print(f"Total gages: {len(gdf)}")
    print(f"\nColumns available: {list(gdf.columns)}")
    print(f"\nFirst few records:")

    # Print available columns
    display_cols = ['site_no', 'station_nm', 'dec_lat_va', 'dec_long_va']
    available_cols = [col for col in display_cols if col in gdf.columns]
    if available_cols:
        print(gdf[available_cols].head())

    print(f"\nGages by state:")
    if 'state_cd' in gdf.columns:
        print(gdf['state_cd'].value_counts().head(20))
