# Setup and usage

[Overview](../README.md) · [Data format](DATA_FORMAT.md) · [Tool guide](../tools/README.md)

## Environment

The core model requires an NVIDIA CUDA environment. Its custom C++/CUDA extension is compiled through `torch.utils.cpp_extension.load` when the model is imported; a CUDA-enabled PyTorch wheel alone does not supply the complete build toolchain. A compatible CUDA toolkit (`nvcc`), C++ compiler, and Ninja are required.

The repository's requirements comments specify Python 3.11, PyTorch 2.1.2, torchvision 0.16.2, CUDA 11.8, and NumPy 1.x. The previous README instead named PyTorch 2.5.1. The commands below follow the requirements file's intended configuration; this cleanup has not verified a fresh CUDA installation.

```bash
conda create -n pcdiff python=3.11 -y
conda activate pcdiff
python -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
nvcc --version
```

For comparison tools and the optional viewers:

```bash
python -m pip install -r requirements-tools.txt
```

Run the following commands from the repository root. Training, inference, and model-dependent validation cannot be fully smoke-tested on a Mac without CUDA, even if `--help` is requested, because imports load the extension.

## Training

Prepare the [dataset layout](DATA_FORMAT.md), then choose resource settings appropriate for your GPU:

```bash
python train_generation.py --path ./data --bs 1 --workers 1 --niter 100
```

The required CLI flags are `--bs`, `--workers`, and `--niter`. The epoch count above is an example. Training writes timestamped run directories under `output/train_generation/`, including `options.txt` and periodic checkpoints.

The current training module sets `CUDA_VISIBLE_DEVICES` at import time. Review that setting before using a cluster scheduler or selecting GPUs; importing the training module from inference/validation can also affect device selection.

## Inference

Supply a compatible checkpoint and keep its `options.txt` beside it when available:

```bash
python infer.py --checkpoint /path/to/epoch.pth --jaw_dir ./data/exampleU --target_fdi 11 --output_dir ./output/inference
```

Use multiple values after `--target_fdi` to select several teeth. `--save_normalized` additionally exports arrays in model coordinates. The CLI currently uses full DDPM sampling; `--ddim_steps` and `--ddim_eta` are accepted but unused.

The CLI does not supply target root-axis features. Interpret axis-conditioned checkpoint outputs with the [architecture notes](ARCHITECTURE.md) in mind.

## Checkpoint validation

```bash
python tools/evaluation/validate.py --checkpoint /path/to/epoch.pth --data_path ./data --num_samples 10
python tools/evaluation/validate_conditioning.py --checkpoint /path/to/epoch.pth --data_path ./data --num_samples 10
```

These scripts serve different experimental settings. Their detailed options and comparison/viewing commands are listed in the [tool guide](../tools/README.md).

## Local assets and source organization

Training data, checkpoints, output folders, machine-specific UI state, and archived source copies are excluded from the source-only publication. An existing local checkout may still contain those files. The separate AxisFlow project is not part of this PCDiff source distribution.
