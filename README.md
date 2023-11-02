![Pytorch](https://img.shields.io/badge/Implemented%20in-Pytorch-red.svg)

![The model architecture](./img/fig11.png)

<!-- <center><b>The Model Struture</b></center><br/>
<center>Source: https://arxiv.org/abs/2011.02881</center>
<br/><br/> -->

This is the codebase for the paper A Two-Stage Generative Model with CycleGAN and Joint Diffusion for MRI-based Brain Tumor Detection. 
This repository is based on [yang-song/score_sde_pytorch](https://github.com/yang-song/score_sde_pytorch), with modifications for conditioning generation.


# Installation

The implementation is based on Python 3.7.13 and Pytorch 1.7.0. You can use the following command to get the environment ready.

```bash
cd path_to_code
pip install -r requirements.txt
```

# Dataset

We trained our method on the [BRATS2020 dataset](https://www.med.upenn.edu/cbica/brats2020/data.html). The input size is padding to (256, 256, 3).


## Folder Structure
  ```
  A-Two-Stage-CycleGAN-VE-BRATS2020/
  │
  ├── main.py - Execute to run in commandline.
  ├── run_lib.py - Runs the training pipeline or Generate samples.
  ├── losses.py - Loss computation and optimization.
  ├── sde_lib.py - Use the reverse-time SDE/ODE.
  ├── sampling.py - Create a sampling function.
  │
  ├── configs/ - holds configuration
  │   ├── ve/ 
  │   ├──├──ncsnpp_continuous.py
  │   ├── vp/ 
  │   ├──├──ddpm_continuous.py
  │   ├── default_configs.py
  │
  ├── dataset - Definition of dataloaders
  │   ├── Brats/
  │   ├──├── Health_png
  │   ├──├── Tumor_png
  │   ├──├── test
  │   └── ...
  │
  ├── models/ - Architecture definitions
  │   ├── ncsnpp.py
  │   ├── ddpm.py
  │   └── ...
  │
  ├── results/ 
  │  
  └── utils/ - small utility functions
      ├── util.py
      └── ...
  ```

# Usage

The Implementations of two stages are separated.


### Training

- For training the first-stage model, follow[CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix):

Train a model:

```bash
python train.py \
--dataroot ./dataset/brats \
--name newbrats --dataset_mode SingleUnaligned --input_nc 1 --output_nc 1 --model cycle_gan --display_id 1 
```

Then transform health to diseased
```bash
python test_single.py \
--dataroot dataset/brats/testB \
--name brats_btoa --dataset_mode bratSingle --model test --input_nc 1 --output_nc 1 --direction BtoA --epoch 30 --model_suffix _B --no_dropout 
```
The result folder is in ./dataset/Brats/Health_png and Tumor_png.

- For training the second-stage conditional VE-JP model, use:

```bash
bash train_brats.sh ve
```

where the `ve` reprents VE-SDE training.

Please change other arguments according to your preference.

### Testing

- For testing the model, use:

```bash
bash test_brats.sh ve
```

where `ve` reprents VE-SDE sampling.


### Benchmarks

1. [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

2. [DPM_classifier](https://github.com/JuliaWolleb/diffusion-anomaly)

3. [DPM_classifier_free](https://github.com/vios-s/Diff-SCM)

4. [f-anogan and VAE](https://github.com/StefanDenn3r/Unsupervised_Anomaly_Detection_Brain_MRI)


Please find more details in our paper and code.