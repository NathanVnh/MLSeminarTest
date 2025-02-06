# A Mixture-Based Framework for Guiding Diffusion Models

[**Paper**](https://arxiv.org/abs/2502.03332) | [**Demo**](https://badr-moufad.github.io/mgdm/)

Code of MGDM algorithm.
Here, we provide the code to run image experiments and audio-source separation.

We also provide the code of the competitors.


## Code installation

### Install project dependencies

Install the code in editable mode by running

```bash
pip install -e .
```

This command will also download the code dependencies.
Further details about dependencies are in ``setup.py`` (or ``pyproject.toml``).

For convenience, the code of these repositories were moved inside ``src`` folder to avoid installation conflicts.

- https://github.com/bahjat-kawar/ddrm
- https://github.com/openai/guided-diffusion
- https://github.com/NVlabs/RED-diff
- https://github.com/mlomnitz/DiffJPEG
- https://github.com/CompVis/latent-diffusion

Also the code of the following repository on which we based our sound was adapted in ``src/sound_model`` module
- https://github.com/gladia-research-group/multi-source-diffusion-models


### Set configuration paths

Since we use the project path for cross referencing, namely open configuration files, ensure to define it in ``src/local_paths.py`` (hint: copy/paste the output of ``pwd`` command)

After [downloading](#downloading-checkpoints) the models checkpoints, make sure to put the corresponding paths in the configuration files.
The fields to change were marked by ``# XXX Change path``

- Model checkoints
  - ``configs/ffhq_model.yaml``
  - ``configs/imagenet_model.yaml``
  - ``configs/ffhq-ldm-vq-4.yaml``
- Nonlinear blur
  - ``src/nonlinear_blurring/option_generate_blur_default.yml``
- Multisource Audio model
  - ``configs/sound_model.yaml``


## Assets

We provide few images of FFHQ and Imagenet.
Some of the degradation operator are also provided as checkpoints to alleviate the initialization overhead.

We also provide one sound Track from Slask2100 dataset.

All this material is located in ``assets/`` folder

```
  assets/
  ├── images/
  ├──── ffhq/
  |       └── im1.png
  |       └── ...
  ├──── imagenet/
  |       └── im1.png
  |       └── ...
  ├──── operators/
  |       └── outpainting_half.pt
  |       └── ...
  ├── Tracks/
  |       └── Track02098
  |           └── ...
```


## Reproduce experiments

We provide two scripts, ``test_images.py`` and ``test_sound.py`` to run the experiments.

### Image restoration tasks

In addition to our algorithm, several state-of-the-art algorithms are supported
``"mgdm"`` (ours), ``"diffpir"``, ``"ddrm"``, ``"ddnm"``, ``"dps"``, ``"pgdm"``, ``"psld"``, ``"reddiff"``, ``"resample"``, ``"daps"`` ``"pnp_dm"``

their hyperparameters are defined in ``configs/experiments/sampler/`` folder for images experiments and ``configs/exp_sound/sampler/`` for audio separation.

we also support several imaging tasks

- **Inpainting**: ``"inpainting_center"``, ``"outpainting_half"``, ``"outpainting_top"``
- **Blurring**: ``"blur"``,  ``"blur_svd"`` (SVD version of blur), ``"motion_blur"``,  ``"nonlinear_blur"``,
- **JPEG dequantization**:  ``"jpeg{QUALITY}"`` (Quality is an integer in [1, 99], example ``"jpeg2"``)
- **Super Resolution**: ``"sr4"``, ``"sr16"``
- **Others**: ``"phase_retrieval"``, ``"high_dynamic_range"``

To run an image experiment, execute the following command

```bash
python test_images.py
```

You can customize the experiments by changing the arguments in the configuration files ``configs/experiments/config.yaml``.

Similarly, run an audio separation experiment by executing

```bash
python test_sound.py
```

and customize it by changing the values of the arguments in ``configs/exp_sound/config.yaml``.


## Downloading checkpoints

- [Imagnet](https://github.com/openai/guided-diffusion)
- [FFHQ](https://github.com/DPS2022/diffusion-posterior-sampling)
- FFHQ LDM: [denoiser](https://ommer-lab.com/files/latent-diffusion/ffhq.zip), [autoencoder](https://ommer-lab.com/files/latent-diffusion/vq-f4.zip)
- [Nonlinear blur operator](https://drive.google.com/file/d/1xUvRmusWa0PaFej1Kxu11Te33v0JvEeL/view?usp=drive_link)
- [Sound Model (MSDM)](https://github.com/gladia-research-group/multi-source-diffusion-models/blob/main/ckpts/README.md)