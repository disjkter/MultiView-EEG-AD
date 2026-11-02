# MultiView-EEG-AD: model code map

Core implementation for multi-view self-supervised EEG representation learning
and participant-level Alzheimer's disease classification.

## Main entry points

- `models/Spatial_MSCNN.py`: spatial and multi-scale temporal encoder branches.
- `scripts/pretrain_tage_timefreq_consistency.py`: waveform and spectral
  reconstruction, cross-branch consistency, and temporal-view consistency.
- `scripts/run_dual_stream_subject_split.py`: supervised fine-tuning and
  participant-level evaluation under train/validation/test participant splits.
- `scripts/run_dual_stream_loso_subject.py`: shared model construction,
  alignment, threshold selection, and evaluation utilities.
- `utils/`: configuration, participant splits, sampling and supporting utilities.
- `configs/` and `config/`: experiment configuration.
- `datasets/` and preprocessing scripts: data-loading and preprocessing code,
  not the EEG datasets themselves.

Other models and experiment scripts include comparison models and research
variants; they are not all components of the main proposed framework.

## Local use

Install dependencies from `requirements.txt` in an appropriate Python/PyTorch
environment. Inspect each entry point with `--help` and configure local data and
checkpoint paths before running. Data, pretrained weights and outputs are not
distributed with the source snapshot. Existing experiment configurations may
refer to the original machine's local paths.

The upstream BioFoundation attribution and license are preserved in README.md
and LICENSE.
