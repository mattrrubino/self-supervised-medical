# Self-Supervised Medical

This project requires installing Python (preferably 3.12) with the `venv` module.
It also assumes you have CUDA installed, with version 11.8 or greater. If this is
not the case, you may need to modify the PyTorch installation in `setup.sh`.

Before running the setup script, you must also create a [Kaggle](https://www.kaggle.com) account and set up
public API usage as described in the [Kaggle documentation](https://www.kaggle.com/docs/api).
You must also join the [APTOS 2019 Blindness Detection Competition](https://www.kaggle.com/competitions/aptos2019-blindness-detection) and accept their terms.
Note that this requires your account to be phone-verified.

Then, to get started, execute the following:

```bash
git clone git@github.com:mattrrubino/self-supervised-medical.git
cd self-supervised-medical
./setup.sh
```

