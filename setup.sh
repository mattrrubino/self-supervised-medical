#!/bin/bash

# Set up the Python environment
python3 -m venv venv
. venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

####################### 3D SETUP #######################

# Download and unzip the data
wget https://msd-for-monai.s3-us-west-2.amazonaws.com/Task07_Pancreas.tar
tar -xf Task07_Pancreas.tar
rm Task07_Pancreas.tar

####################### 2D SETUP #######################

# Exit on error
set -e

echo "Setting up environment for APTOS 2019 Blindness Detection..."

# Ensure pip and kaggle package are installed
if ! command -v kaggle &> /dev/null; then
    echo "Installing kaggle CLI via pip..."
    pip install --user kaggle
    export PATH="$HOME/Library/Python/3.*/bin:$PATH"
fi

# Set up Kaggle API credentials
KAGGLE_DIR="$HOME/.kaggle"
KAGGLE_JSON="$KAGGLE_DIR/kaggle.json"

echo $KAGGLE_DIR

# If the ~/.kaggle directory doesn't exist, create it
mkdir -p "$KAGGLE_DIR"

# Move kaggle.json into place if it exists in the current directory
if [ -f "./kaggle.json" ]; then
    echo "Found kaggle.json in current directory. Moving it to $KAGGLE_JSON..."
    mv ./kaggle.json "$KAGGLE_JSON"
fi

# Ensure the kaggle.json file is in place
if [ ! -f "$KAGGLE_JSON" ]; then
    echo "kaggle.json not found. Please place your API token in either the current directory or $KAGGLE_JSON"
    exit 1
fi

chmod 600 "$KAGGLE_JSON"

# Make directory and download dataset
DATASET_DIR="src/2d/dataset"
mkdir -p "$DATASET_DIR"
cd "$DATASET_DIR"

echo "Downloading dataset from Kaggle..."
kaggle competitions download -c aptos2019-blindness-detection

echo "Extracting dataset..."
unzip -q '*.zip'

echo "Setup complete! Dataset is located in: $(pwd)"