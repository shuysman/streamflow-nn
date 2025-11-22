library(nhdplusTools)
library(sf)
library(mapview)

# 1. Define the "Starting Points" (Outlets)
# We use USGS Gauge IDs as the most reliable anchor points.
# Lamar River near Tower Falls, WY: USGS-06191500
# Hoh River at US Hwy 101 near Forks, WA: USGS-12041200
nldi_feature_lamar <- list(featureSource = "nwissite", featureID = "USGS-06191500")
nldi_feature_hoh   <- list(featureSource = "nwissite", featureID = "USGS-12041200")

# 2. Retrieve the Total Upstream Basin (The "Watershed Polygon")
# This function traces flowlines upstream and merges the boundaries.
lamar_basin <- get_nldi_basin(nldi_feature = nldi_feature_lamar)
hoh_basin   <- get_nldi_basin(nldi_feature = nldi_feature_hoh)

# 3. (Optional) Retrieve the Constituent Catchments
# Since you are doing distributed modeling with gridMET, you might want the 
# individual sub-catchments inside the basin rather than just the outline.
# This grabs all upstream flowlines (UT) and then fetches their catchments.
lamar_flowlines <- navigate_nldi(nldi_feature_lamar, mode = "upstreamTributaries", distance_km = 1000)
lamar_catchments <- subset_nhdplus(comids = lamar_flowlines$UT_flowlines$nhdplus_comid, 
                                   nhdplus_data = "download", 
                                   flowline_only = FALSE) # Set to FALSE to get polygons

# 4. Visualize to verify
mapview(lamar_basin, col.regions = "blue", alpha.regions = 0.2, layer.name = "Lamar Basin") +
  mapview(hoh_basin, col.regions = "green", alpha.regions = 0.2, layer.name = "Hoh Basin")
