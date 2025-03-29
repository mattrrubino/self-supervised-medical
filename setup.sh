#!/bin/bash

# Set up the Python environment
python3 -m venv venv
. venv/bin/activate
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# Download and unzip the data
wget https://msd-for-monai.s3-us-west-2.amazonaws.com/Task07_Pancreas.tar
tar -xf Task07_Pancreas.tar
rm Task07_Pancreas.tar

