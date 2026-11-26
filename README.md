# PCDiff-CBCT · Dental Point-Cloud Generation

**Diffusion-based reconstruction of missing-tooth geometry from CBCT-derived tooth point clouds.**

A research project exploring dental geometry generation for implant-planning research. The implementation brings together tooth-position embeddings, missing-tooth masks, point–voxel neural networks, diffusion sampling, and root-axis conditioning experiments.

[Architecture](docs/ARCHITECTURE.md) · [Setup and usage](docs/SETUP.md) · [Data format](docs/DATA_FORMAT.md) · [Evaluation and visualization tools](tools/README.md)

## What the project implements

| Component | Implementation |
| --- | --- |
| **Dental data representation** | Per-tooth point clouds arranged into 28 FDI-indexed slots, with explicit masks for observed, selected, and naturally absent teeth. |
| **Diffusion training** | Noise-prediction training with a configurable diffusion schedule, masked losses, checkpointing, and distributed GPU training. |
| **Point–voxel backbone** | PVCNN-style point/voxel feature processing, PointNet set abstraction and feature propagation, attention, and custom CUDA operations. |
| **Tooth identity and axis features** | Learned FDI-position embeddings, observation/existence indicators, and optional root-axis features for compatible checkpoints. |
| **Missing-tooth inference** | Select one or more target FDI numbers, sample generated point clouds from a checkpoint, and export NumPy arrays. |
| **Evaluation and inspection** | Point-cloud distance and root-axis comparison utilities, batch CSV summaries, and interactive dental visualizations. |

## Workflow

1. **Prepare tooth point clouds.** Supply a jaw folder containing one `.npy` point cloud per tooth, optionally accompanied by precomputed root-axis information.
2. **Build the training representation.** The loader samples points, applies the repository's fixed normalization, assigns FDI embeddings, and records missing-tooth masks.
3. **Train the denoiser.** The training pipeline simulates missing teeth and learns to predict added noise using the configured tooth features.
4. **Generate selected teeth.** The inference CLI selects target FDI positions and runs reverse diffusion using a compatible checkpoint.
5. **Inspect outputs.** Compare generated geometry with reference point clouds or visualize it alongside the observed dentition.

## Repository layout

```text
pcdiff_cbct/
├── train_generation.py       # Training, diffusion process, and model wrapper
├── infer.py                  # Missing-tooth inference CLI
├── datasets/                 # Tooth loading, normalization, and augmentation helpers
├── model/                    # Point–voxel backbone and supporting encoders
├── modules/                  # Neural-network layers and CUDA extension sources
├── utils/                    # Logging, output management, and export helpers
├── tools/
│   ├── evaluation/           # Checkpoint validation and point-cloud comparisons
│   └── visualization/        # Dental geometry and root-axis viewers
├── docs/                     # Setup, data contract, and architecture
└── requirements.txt          # Core Python dependencies
```

`train_generation.py` and `infer.py` remain at the root as the primary entry points. Datasets, checkpoints, generated outputs, and duplicate source archives are separate from the source distribution.

## Technology stack

**Python · PyTorch · CUDA/C++ · NumPy · Open3D · Matplotlib**

Evaluation tools additionally use scikit-learn; selected viewers use SciPy or Polyscope. See [setup](docs/SETUP.md) for the repository's dependency configuration and hardware requirements.

## Getting started

```bash
python train_generation.py --path ./data --bs 1 --workers 1 --niter 100
python infer.py --checkpoint /path/to/epoch.pth --jaw_dir ./data/exampleU --target_fdi 11 --output_dir ./output/inference
```

These are example invocations, not prescribed training settings. Install the CUDA environment, supply compatible data and checkpoints, and review the [setup guide](docs/SETUP.md) first.

## Research scope

The input is already prepared tooth geometry; this repository does not provide a complete raw-CBCT segmentation or implant-placement workflow. Generated outputs are tooth point clouds.

The current backbone processes tooth slots independently after reshaping them into the batch dimension. Its attention operates within a tooth; explicit cross-tooth context modeling is not implemented in this version. Axis-conditioned validation and inference without target-axis information are different experimental settings. These boundaries are detailed in the [architecture guide](docs/ARCHITECTURE.md).

This overview describes the implementation and does not claim clinical validation or report performance results.
