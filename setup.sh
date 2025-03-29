#!/bin/bash

# Set up the Python environment
python3 -m venv venv
. venv/bin/activate
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# Download the data
kaggle competitions download -c aptos2019-blindness-detection

