# Architecture

[Overview](../README.md) · [Setup](SETUP.md) · [Data format](DATA_FORMAT.md)

## Data and masks

[ToothDataset](../datasets/dataset_generation.py) loads upper/lower jaw folders and places sampled tooth point clouds into 28 FDI slots. The default representation has 1,024 points per tooth. FDI indices identify anatomical positions, while an independent mask identifies naturally absent teeth.

Training creates observed and latent masks for simulated missing teeth. The loss is applied to selected latent teeth. This separates teeth selected for reconstruction from teeth already absent in the source data.

## Denoising model

[PVCNN2Base](../model/pvcnn_generation.py) combines point–voxel convolution, PointNet abstraction/propagation, time embeddings, tooth-position embeddings, and attention. [modules/](../modules/) includes the neural-network building blocks and CUDA kernels for neighborhood queries, sampling, grouping, interpolation, and voxel operations.

The per-point feature channels combine an eight-dimensional FDI embedding, an observation indicator, an existence indicator, and optionally a three-dimensional root axis. Checkpoint architectures differ between the 10- and 13-extra-feature configurations.

Although inputs are arranged as a full set of tooth slots, the backbone reshapes them to `(batch × teeth, channels, points)`. Its current attention therefore operates within a tooth. The presence of a layout-transformer implementation in [bounding_encoder.py](../model/bounding_encoder.py) should not be interpreted as active cross-tooth conditioning in this training path.

## Diffusion and training

[train_generation.py](../train_generation.py) contains the diffusion schedules, noise-prediction objective, reverse-sampling methods, model wrapper, and distributed training loop. Training uses PyTorch distributed processes across visible CUDA devices and writes checkpoints alongside run options and logs.

Both DDPM and DDIM sampling methods exist in the model wrapper. The [inference CLI](../infer.py) explicitly requests DDPM; its DDIM-related command-line arguments are currently marked unused.

## Root-axis experiments

The dataset loads precomputed root-axis dictionaries when available and otherwise computes fallback axis information. Training and internal validation can pass axis vectors for all teeth, including simulated targets.

That setting differs from reconstructing a genuinely missing tooth whose axis is unknown. The inference CLI does not pass root-axis vectors, so the axis-enabled backbone receives zero-valued axis features in that path. [validate_conditioning.py](../tools/evaluation/validate_conditioning.py) provides a separate validation workflow for conditioning experiments. These modes should be kept distinct when interpreting outputs.

## Evaluation and visualization

The [evaluation tools](../tools/README.md) compare shapes, centroids, and PCA-derived axes. The EMD-named helper uses greedy matching rather than an exact optimal-transport solver. Several comparison paths center each cloud independently by default; use their `--no_center` option when the intended comparison must retain translation differences.

Visualization tools inspect context teeth, generated geometry, references, and root-axis directions. They use different coordinate assumptions; consult each tool's arguments before mixing normalized and denormalized arrays.

## Implementation boundaries

- Data loading uses a sorted 80/20 split of jaw folders, not an explicit patient-grouped split.
- Dataset augmentation hooks exist, but the augmentation block in the current loader is commented out.
- Coordinates use fixed normalization constants; appropriate registration and unit consistency are input requirements.
- Importing the model triggers compilation/loading of custom CUDA extensions.
- The core implementation is a generation research pipeline; raw-volume segmentation and downstream clinical planning are outside this code's scope.
