# MScThesis

## Getting started

To clone the repository, run the following command:

```sh
git clone https://github.com/JvanZyl1/MScThesis.git
```

To create a new conda environment using the `environment.yml` file, run the following command:

```sh
conda env create --name new_env_name -f [environment.yml]
```

Check that torch is properly configured for CUDA, through the lines:
```py
import torch
print("CUDA available:", torch.cuda.is_available())
print("GPU device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
```