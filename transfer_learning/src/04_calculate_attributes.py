#!/usr/bin/env python3
"""
Stage 1.4: Calculate Static Attributes

Compute CAMELS-style static attributes for each watershed from geospatial data:
- elev_mean: Mean elevation from USGS 3DEP 30m DEM
- slope_mean: Mean slope derived from 3DEP DEM gradient
- area_gages2: Drainage area from USGS gage metadata (authoritative)
- frac_forest: Forest fraction from NLCD 2021 (classes 41, 42, 43)
- soil_depth_pelletier: Soil depth from NRCS SSURGO (component-weighted max horizon depth)
- sand_frac: Sand fraction from NRCS SSURGO (component-weighted average)
- clay_frac: Clay fraction from NRCS SSURGO (component-weighted average)

Downloaded rasters are cached to ~/sync/dem_data/

Usage:
    python src/04_calculate_attributes.py --watershed lamar
    python src/04_calculate_attributes.py --watershed hoh
    python src/04_calculate_attributes.py --watershed lamar --no-cache
"""

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.mask
import requests
from pygeoogc import WMS
from shapely.geometry import mapping

# Authoritative USGS drainage areas (km²) — also used in streamflow unit conversion
USGS_AREAS = {
    'lamar': 1730.1,  # USGS 06188000: 668.0 sq mi
    'hoh': 655.3,     # USGS 12041200: 253.0 sq mi
}

WATERSHED_GEOJSON = {
    'lamar': Path(__file__).parent.parent.parent / 'data' / 'watershed' / 'lamar_watershed.geojson',
    'hoh': Path(__file__).parent.parent.parent / 'data' / 'watershed' / 'hoh_watershed.geojson',
}

CACHE_DIR = Path.home() / 'sync' / 'dem_data'

# WMS endpoints
DEM_WMS_URL = 'https://elevation.nationalmap.gov/arcgis/services/3DEPElevation/ImageServer/WMSServer'
DEM_LAYER = '3DEPElevation:None'
NLCD_WMS_URL = 'https://www.mrlc.gov/geoserver/mrlc_download/wms'
NLCD_LAYER = 'NLCD_2021_Land_Cover_L48'

# SSURGO Soil Data Access API
SDA_URL = 'https://SDMDataAccess.sc.egov.usda.gov/Tabular/SDMTabularService/post.rest'


def load_watershed(name):
    """Load watershed boundary GeoDataFrame."""
    path = WATERSHED_GEOJSON[name]
    gdf = gpd.read_file(path)
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.set_crs('EPSG:4326')
    return gdf


def _fetch_wms_tif(watershed, gdf, cache_name, wms_url, layer):
    """Fetch a WMS layer as GeoTIFF, with caching.

    Re-saves with explicit georeferencing since some WMS servers return
    TIFFs without proper geotransform metadata.
    """
    cache_path = CACHE_DIR / f'{watershed}_{cache_name}.tif'
    if cache_path.exists():
        print(f"  Using cached {cache_path.name}")
        return cache_path

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    bounds = gdf.total_bounds
    bbox = (float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3]))
    print(f"  Fetching {layer} for bbox {[f'{x:.3f}' for x in bbox]}...")

    wms = WMS(wms_url, layers=layer, outformat='image/tiff', crs=4326)
    data = wms.getmap_bybox(bbox, resolution=30, box_crs=4326)
    tiff_bytes = list(data.values())[0]

    # Re-save with explicit georeferencing
    import io
    with rasterio.open(io.BytesIO(tiff_bytes)) as src:
        arr = src.read()
        height, width = arr.shape[1], arr.shape[2]
        # Build transform from bbox and array dimensions
        transform = rasterio.transform.from_bounds(*bbox, width, height)
        profile = src.profile.copy()
        profile.update(crs='EPSG:4326', transform=transform)

        with rasterio.open(cache_path, 'w', **profile) as dst:
            dst.write(arr)

    size_mb = cache_path.stat().st_size / 1e6
    print(f"  Saved {cache_path.name} ({size_mb:.1f} MB)")
    return cache_path


def _clip_raster(tif_path, gdf):
    """Clip a raster to watershed boundary, return masked array.

    For float rasters, nodata pixels are set to NaN.
    For integer rasters (e.g., NLCD), nodata pixels are set to 0.
    """
    with rasterio.open(tif_path) as src:
        geom = [mapping(gdf.geometry.values[0])]
        is_int = np.issubdtype(src.dtypes[0], np.integer)
        fill_val = 0 if is_int else np.nan
        out_image, _ = rasterio.mask.mask(src, geom, crop=True, nodata=fill_val)
        arr = out_image[0].astype(np.float32) if not is_int else out_image[0]
        nodata = src.nodata
        if nodata is not None and not is_int:
            arr = np.where(arr == nodata, np.nan, arr)
        return arr, src


def fetch_dem(watershed, gdf):
    """Fetch 3DEP 30m DEM."""
    return _fetch_wms_tif(watershed, gdf, 'dem_30m', DEM_WMS_URL, DEM_LAYER)


def compute_elev_mean(dem_path, gdf):
    """Compute mean elevation within watershed."""
    arr, _ = _clip_raster(dem_path, gdf)
    return float(np.nanmean(arr))


def compute_slope_mean(dem_path, gdf):
    """Compute mean slope (degrees) from DEM gradient within watershed."""
    arr, _ = _clip_raster(dem_path, gdf)

    # Pixel size in meters (approximate at watershed center latitude)
    with rasterio.open(dem_path) as ds:
        res_x = abs(ds.transform.a)  # degrees per pixel
        res_y = abs(ds.transform.e)

    center_lat = float(gdf.to_crs('EPSG:5070').geometry.centroid.to_crs('EPSG:4326').y.values[0])
    meters_per_deg_lat = 111320.0
    meters_per_deg_lon = 111320.0 * np.cos(np.radians(center_lat))

    dx = res_x * meters_per_deg_lon
    dy = res_y * meters_per_deg_lat

    # Gradient in m/m
    dzdx = np.gradient(arr, dx, axis=1)
    dzdy = np.gradient(arr, dy, axis=0)
    slope_rad = np.arctan(np.sqrt(dzdx**2 + dzdy**2))
    slope_deg = np.degrees(slope_rad)

    return float(np.nanmean(slope_deg))


def fetch_nlcd(watershed, gdf):
    """Fetch NLCD 2021 land cover."""
    return _fetch_wms_tif(watershed, gdf, 'nlcd_2021', NLCD_WMS_URL, NLCD_LAYER)


def compute_frac_forest(nlcd_path, gdf):
    """Compute forest fraction from NLCD (classes 41=deciduous, 42=evergreen, 43=mixed)."""
    arr, _ = _clip_raster(nlcd_path, gdf)
    valid = ~np.isnan(arr) & (arr > 0)
    forest = np.isin(arr, [41, 42, 43]) & valid
    return float(forest.sum() / valid.sum())


def fetch_ssurgo(watershed, gdf):
    """Query NRCS SSURGO for soil properties via Soil Data Access REST API.

    Returns dict with sand_frac, clay_frac, soil_depth_pelletier (in meters).
    """
    cache_path = CACHE_DIR / f'{watershed}_ssurgo.json'
    if cache_path.exists():
        print(f"  Using cached {cache_path.name}")
        with open(cache_path) as f:
            return json.load(f)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Simplify polygon for the SQL query (reduce WKT size)
    simple = gdf.geometry.simplify(0.01).values[0]
    wkt = simple.wkt

    query = f"""
    SELECT
        c.comppct_r,
        AVG(ch.sandtotal_r) as sand_avg,
        AVG(ch.claytotal_r) as clay_avg,
        MAX(ch.hzdepb_r) as max_depth_cm
    FROM SDA_Get_Mukey_from_intersection_with_WktWgs84('{wkt}') AS mu
    INNER JOIN component AS c ON c.mukey = mu.mukey
    INNER JOIN chorizon AS ch ON ch.cokey = c.cokey
    WHERE c.comppct_r > 0
      AND ch.sandtotal_r IS NOT NULL
    GROUP BY c.cokey, c.comppct_r
    """

    print(f"  Querying SSURGO Soil Data Access API...")
    resp = requests.post(SDA_URL, json={'query': query, 'format': 'JSON+COLUMNNAME+METADATA'},
                         timeout=120)
    resp.raise_for_status()
    result = resp.json()

    if 'Table' not in result:
        raise RuntimeError(f"SSURGO query returned no data: {result}")

    rows = result['Table']
    # rows[0] = headers, rows[1] = metadata, rows[2:] = data
    data_rows = rows[2:]
    print(f"  Got {len(data_rows)} soil component records")

    total_weight = 0.0
    sand_sum = clay_sum = depth_sum = 0.0
    for row in data_rows:
        weight = float(row[0])
        sand = float(row[1])
        clay = float(row[2])
        depth_cm = float(row[3])
        total_weight += weight
        sand_sum += weight * sand
        clay_sum += weight * clay
        depth_sum += weight * depth_cm

    soil = {
        'sand_frac': round(sand_sum / total_weight / 100, 4),
        'clay_frac': round(clay_sum / total_weight / 100, 4),
        'soil_depth_pelletier': round(depth_sum / total_weight / 100, 2),
    }

    with open(cache_path, 'w') as f:
        json.dump(soil, f, indent=2)
    print(f"  Saved {cache_path.name}")

    return soil


def calculate_attributes(watershed, output_file):
    """Compute all static attributes for a watershed and write CSV."""
    print(f"\n{'='*60}")
    print(f"Computing Attributes: {watershed.upper()}")
    print(f"{'='*60}")

    gdf = load_watershed(watershed)

    # Area from USGS gage metadata
    area = USGS_AREAS[watershed]
    print(f"\nDrainage area (USGS): {area} km²")

    # Elevation and slope from 3DEP DEM
    print(f"\nFetching 3DEP DEM (30m)...")
    dem_path = fetch_dem(watershed, gdf)
    elev_mean = compute_elev_mean(dem_path, gdf)
    slope_mean = compute_slope_mean(dem_path, gdf)
    print(f"  elev_mean: {elev_mean:.1f} m")
    print(f"  slope_mean: {slope_mean:.1f} degrees")

    # Forest fraction from NLCD
    print(f"\nFetching NLCD 2021...")
    nlcd_path = fetch_nlcd(watershed, gdf)
    frac_forest = compute_frac_forest(nlcd_path, gdf)
    print(f"  frac_forest: {frac_forest:.3f} ({frac_forest*100:.1f}%)")

    # Soil properties from SSURGO
    print(f"\nFetching SSURGO soil data...")
    soil = fetch_ssurgo(watershed, gdf)
    print(f"  soil_depth_pelletier: {soil['soil_depth_pelletier']} m")
    print(f"  sand_frac: {soil['sand_frac']} ({soil['sand_frac']*100:.1f}%)")
    print(f"  clay_frac: {soil['clay_frac']} ({soil['clay_frac']*100:.1f}%)")

    # Assemble DataFrame
    attrs = {
        'basin': f'{watershed}_lumped',
        'elev_mean': round(elev_mean, 1),
        'slope_mean': round(slope_mean, 1),
        'area_gages2': area,
        'frac_forest': round(frac_forest, 3),
        'soil_depth_pelletier': soil['soil_depth_pelletier'],
        'sand_frac': soil['sand_frac'],
        'clay_frac': soil['clay_frac'],
    }

    df = pd.DataFrame([attrs])

    # Validation
    print(f"\n=== Validation ===")
    assert 0 < attrs['elev_mean'] < 5000, f"Elevation out of range: {attrs['elev_mean']}"
    assert 0 <= attrs['slope_mean'] < 90, f"Slope out of range: {attrs['slope_mean']}"
    assert attrs['area_gages2'] > 0, "Drainage area must be positive"
    assert 0 <= attrs['frac_forest'] <= 1, f"Forest fraction out of range: {attrs['frac_forest']}"
    assert attrs['soil_depth_pelletier'] > 0, f"Soil depth must be positive"
    assert 0 <= attrs['sand_frac'] <= 1, f"Sand fraction out of range"
    assert 0 <= attrs['clay_frac'] <= 1, f"Clay fraction out of range"
    frac_sum = attrs['sand_frac'] + attrs['clay_frac']
    assert frac_sum <= 1.0, f"Sand + clay fractions exceed 1.0: {frac_sum}"
    print(f"All values in valid ranges")

    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)

    print(f"\nSaved: {output_file}")
    print(f"\n=== CSV Content ===")
    print(df.to_string(index=False))

    return df


def main():
    parser = argparse.ArgumentParser(description='Calculate static attributes for watershed')
    parser.add_argument('--watershed', required=True, choices=['lamar', 'hoh'],
                        help='Watershed name')
    parser.add_argument('--no-cache', action='store_true',
                        help='Force re-download of all data (ignore cache)')

    args = parser.parse_args()

    if args.no_cache:
        # Remove cached files for this watershed
        for f in CACHE_DIR.glob(f'{args.watershed}_*'):
            f.unlink()
            print(f"Removed cached: {f.name}")

    base_dir = Path(__file__).parent.parent
    output_file = base_dir / f'data/neuralhydrology/attributes/{args.watershed}_attributes.csv'

    calculate_attributes(args.watershed, output_file)


if __name__ == '__main__':
    main()
