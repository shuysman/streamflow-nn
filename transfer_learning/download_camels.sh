#!/bin/bash
# Script to download CAMELS US data from Zenodo mirror
# Total size: ~15GB for the main archive

BASE_URL="https://zenodo.org/records/15529996/files"
DATA_DIR="transfer_learning/data/CAMELS_US"

mkdir -p "$DATA_DIR/camels_attributes_v2.0"

echo "--- Downloading Attributes ---"
ATTRIBUTES=("camels_clim.txt" "camels_geol.txt" "camels_hydro.txt" "camels_name.txt" "camels_soil.txt" "camels_topo.txt" "camels_vege.txt")

for file in "${ATTRIBUTES[@]}"; do
    if [ ! -f "$DATA_DIR/camels_attributes_v2.0/$file" ]; then
        echo "Fetching $file..."
        wget -q --show-progress -O "$DATA_DIR/camels_attributes_v2.0/$file" "$BASE_URL/$file"
    else
        echo "$file already exists."
    fi
done

echo "--- Downloading Time Series (Forcing & Flow) ---"
echo "Note: This is a large file (approx 15GB)."
ZIP_FILE="basin_timeseries_v1p2_metForcing_obsFlow.zip"

if [ ! -f "$DATA_DIR/$ZIP_FILE" ]; then
    read -p "Would you like to start the 15GB download now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        wget --show-progress -O "$DATA_DIR/$ZIP_FILE" "$BASE_URL/$ZIP_FILE"
        echo "Extracting... (this may take a while)"
        unzip -q "$DATA_DIR/$ZIP_FILE" -d "$DATA_DIR"
        # The zip contains a folder 'basin_dataset_public_v1p2'
        # We need to map it to the structure NeuralHydrology expects
        mv "$DATA_DIR/basin_dataset_public_v1p2/basin_mean_forcing" "$DATA_DIR/"
        mv "$DATA_DIR/basin_dataset_public_v1p2/usgs_streamflow" "$DATA_DIR/"
        echo "Extraction complete."
    fi
else
    echo "Time series zip already exists."
fi

echo "--- Done ---"
