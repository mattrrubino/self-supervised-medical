# Self-Supervised Medical

This project requires installing Python (preferably 3.12) with the `venv` module.
It also assumes you have CUDA installed, with version 11.8 or greater. If this is
not the case, you may need to modify the PyTorch installation in `setup.sh`.

To start, you must configure your Kaggle account to download the APTOS 2019 Blindness Detection dataset:

1) Ensure you have a Kaggle account created: https://www.kaggle.com/
2) Create a Kaggle API token. To do this, navigate to your Kaggle Settings (https://www.kaggle.com/settings), scroll down, and click "Create New Token". This will download a kagge.json file which you need to move into the root of this repository.
3) Navigate to the APTOS 2019 Blindness Detection Kaggle page (https://www.kaggle.com/competitions/aptos2019-blindness-detection/data), scroll down, and agree to the competition rules. You must do this before Kaggle will let you download the data.

Then, execute the following:

```bash
git clone git@github.com:mattrrubino/self-supervised-medical.git
cd self-supervised-medical
./code/setup.sh
```
