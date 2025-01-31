# This file meant to retain compatibility with older version of python
# when installing the package in edit mode

from setuptools import setup


requirements = [
    "torch",
    "torchvision",
    "transformers",
    "diffusers[torch]",
    "lpips",
    "Pillow",
    "numpy",
    "scikit-learn",
    "matplotlib",
    "tqdm",
    "omegaconf",
    "tqdm",
    "PyYAML",
    "einops",
    "lightning",
    "taming-transformers-rom1504",
    "hydra-core",
    "pytorch_fid",
    # --- For audio model
    "torchaudio",
    "audio-diffusion-pytorch==0.0.43",  # install this version to align with MDSM
    "librosa",
    "av",
    "ipython",
    # ---
]


setup(
    name="mgdm",
    version="0.0.0",
    install_requires=requirements,
)
