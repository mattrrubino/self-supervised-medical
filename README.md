# Introdcution
This project aims to reimplement the Paper ["3D Self-Supervised Methods for Medical Imaging"](https://arxiv.org/abs/2006.03829) [(original repo)](https://github.com/HealthML/self-supervised-3d-tasks) where we pretrain on two separate 2D classification and 3D segmentation tasks.  This paper demonstrates that 3D pretext tasks substantially improve downstream segmentation and classification accuracy, especially in small data regimes. Their key contributions are: 1) Formulation of five 3D SSL tasks, 2) Open‑source implementations of SSL tasks, and 3) Comprehensive evaluations on pancreas tumor segmentation, brain tumor segmentation, and diabetic retinopathy classification, empirically demonstrating efficiency gains produced by SSL.



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
source venv/bin/activate
```

# Running 2D Case
## Pretraining
To pretrain the 2D models, first navigate to code/2d/ directory. Simply run:

```
python train2D.py
```

Follow the prompting intructions on the CLI to see what pretext tasks you wish to run

## Finetuning
To finetune the 2d moels, first navigate to code/2d/ directory. First open the file `finetune.py` and change checkpoint files (if nessecary) in `reset_model_weights` function. Then simply run:

```
python finetune.py
```

Follow the CLI prompts for which task you wish to finetune for. 







