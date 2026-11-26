# Evaluation and visualization tools

[Overview](../README.md) · [Setup](../docs/SETUP.md)

Run commands from the repository root after installing the project dependencies and `requirements-tools.txt`. The validation scripts locate project imports relative to their own file; CLI input/output paths remain relative to your working directory.

## Evaluation

| Tool | Purpose |
| --- | --- |
| [validate.py](evaluation/validate.py) | Generate validation samples from a checkpoint and compute comparison metrics |
| [validate_conditioning.py](evaluation/validate_conditioning.py) | Validate root-axis conditioning experiments |
| [compare_pointclouds.py](evaluation/compare_pointclouds.py) | Compare two supplied `.npy` point clouds |
| [batch_compare.py](evaluation/batch_compare.py) | Compare generated/reference pairs in sample folders and export CSV |

```bash
python tools/evaluation/compare_pointclouds.py generated.npy reference.npy --no_center
python tools/evaluation/batch_compare.py /path/to/samples --out ./comparison.csv
```

The EMD-named metric is a greedy matching approximation. Comparison tools center clouds independently by default; pass `--no_center` to retain positional offsets.

## Visualization

| Tool | Purpose |
| --- | --- |
| [visualize_dental_gen.py](visualization/visualize_dental_gen.py) | View generated teeth beside a jaw's context and references |
| [visualize_gen_folder.py](visualization/visualize_gen_folder.py) | Inspect a generated tooth with denormalization and pose adjustments |
| [visualize_generated_sample.py](visualization/visualize_generated_sample.py) | View exported normalized context/generated/reference arrays |
| [visualize_root_axes.py](visualization/visualize_root_axes.py) | Inspect tooth geometry and saved root axes |
| [visualize_ps.py](visualization/visualize_ps.py) | Interactive Polyscope point-cloud viewer |

```bash
python tools/visualization/visualize_dental_gen.py --patient_folder ./data/exampleU --target_fdi 11 --generated ./output/tooth.npy
python tools/visualization/visualize_root_axes.py --patient_folder ./data/exampleU --fdi 11
python tools/visualization/visualize_ps.py context.npy generated.npy --axes
```

Interactive viewers require a graphical desktop. Coordinate assumptions differ by viewer: inspect its help before combining normalized and denormalized arrays. The Polyscope viewer now defaults to CLI file arguments; its optional `FILES` and `LABELS` overrides are empty.

## Older experiments

[visualize_normalize.py](experiments/visualize_normalize.py) and [visualize_single_npy.py](experiments/visualize_single_npy.py) are preserved as manually configured experiments. They contain machine-specific paths and are not general-purpose CLIs.
