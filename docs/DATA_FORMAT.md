# Data format

[Overview](../README.md) · [Setup](SETUP.md)

The core pipeline consumes prepared per-tooth point clouds. Dataset acquisition, raw CBCT segmentation, and conversion into this layout are not implemented as one integrated workflow here.

## Directory layout

```text
data/
├── exampleU/
│   ├── verts/
│   │   ├── tooth_FDI_11.npy
│   │   └── tooth_FDI_12.npy
│   └── root_axes/                 # Optional precomputed axis dictionaries
│       ├── tooth_FDI_11_root_axis.npy
│       └── tooth_FDI_12_root_axis.npy
└── exampleL/
    └── verts/
        └── tooth_FDI_41.npy
```

`exampleU` and `exampleL` are illustrative folder names. The training loader selects folders containing `U` or `L`. Use filenames ending in `_FDI_XX.npy` for compatibility with both training and inference.

## Arrays and identity

- Each tooth file contains a numeric `(N, 3)` array of XYZ coordinates.
- The loader samples points to the configured `tooth_npoints` count, which defaults to 1,024.
- Supported positions are FDI 11–17, 21–27, 31–37, and 41–47. Wisdom teeth are not included in the current 28-slot list.
- Missing slots are represented with noise and a separate absence mask, rather than a special tooth-identity embedding.
- Optional root-axis files contain dictionaries consumed by the loader. Consult [dataset_generation.py](../datasets/dataset_generation.py) for the exact fields and fallback calculation.

## Normalization and splits

Training and inference use the fixed shift `[-0.72, 25.35, -6.64]` and scale `39.2`. Incoming point clouds must follow the coordinate registration and scale expected by the checkpoint. These constants are not a substitute for validating the units of a new dataset.

The loader sorts jaw folders and assigns the first 80% to training and the remainder to validation. It does not enforce separation of two jaws belonging to the same patient.

The source repository does not supply a dataset download or a checkpoint download. Keep data and weights local and provide their paths explicitly.
