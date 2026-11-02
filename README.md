# MultiView-EEG-AD

Official research code for:

> **Multi-View Self-Supervised EEG Representation Learning Framework for Alzheimer's Disease Classification**<br>
> Zhengyang Liang, School of Artificial Intelligence, South China Normal University

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

## Overview

This repository implements a label-efficient framework for binary Alzheimer's
disease (AD) versus healthy control (HC) classification from EEG. The method
first learns from unlabeled EEG through multi-view self-supervised pretraining,
then transfers the encoder to participant-level supervised fine-tuning.

The framework contains three main ideas:

1. **Dual-view EEG encoder.** A spatial branch models cross-channel
   interactions, while a temporal branch learns multi-scale temporal patterns.
2. **Complementary self-supervised objectives.** Waveform reconstruction,
   spectral reconstruction, cross-branch consistency, and temporal-view
   consistency jointly guide representation learning.
3. **Participant-level label-efficient evaluation.** Label budgets count
   participants rather than repeated EEG windows. Data splits, validation-based
   threshold selection, and final prediction are all participant-aware.

The standard input is a 1-second EEG window with 19 channels sampled at 128 Hz,
represented as `19 x 128`. Each encoder branch produces a 256-dimensional
feature, and concatenation forms a 512-dimensional representation.

## Framework

```text
Unlabeled EEG
    |
    +-- Hybrid channel/time masking
    |
    +-- Spatial view: cross-channel modeling ---------+
    |                                                 +--> 512-D representation
    +-- Temporal view: multi-scale temporal modeling -+
                              |
                              +-- waveform reconstruction
                              +-- spectral reconstruction
                              +-- cross-branch consistency
                              +-- temporal-view consistency

Pretrained encoder
    |
    +-- participant-level supervised fine-tuning
    +-- window probability aggregation
    +-- validation-selected decision threshold
    +-- participant prediction: AD / HC
```

## Results Reported in the Manuscript

All results use participant-level separation between training, validation, and
test sets.

| Dataset | Accuracy | Balanced accuracy | F1 | AUROC |
|---|---:|---:|---:|---:|
| CAUEEG | 0.879 | 0.870 | 0.823 | 0.929 |
| ds004504 | 0.818 | 0.810 | 0.859 | 0.842 |

In the four-participants-per-class setting on CAUEEG, self-supervised
initialization improves AUROC from `0.603` to `0.717`. Please consult the paper
for the complete protocols, seed-level statistics, ablations, and limitations.

## Repository Structure

```text
models/
  Spatial_MSCNN.py                    # dual-view spatial-temporal encoder
scripts/
  preprocess_caueeg.py                # CAUEEG preprocessing
  preprocess_ds004504.py              # ds004504 preprocessing
  pretrain_tage_timefreq_consistency.py
  run_dual_stream_subject_split.py     # participant-level fine-tuning
  run_dual_stream_loso_subject.py      # shared evaluation utilities
configs/                               # experiment configurations
utils/                                 # splits, sampling, and configuration helpers
tests/                                 # unit and protocol checks
```

The repository also contains comparison models and research variants used in
the experimental study. The files listed above are the main entry points for
the proposed framework.

## Installation

Clone the repository and create a Python environment:

```bash
git clone https://github.com/disjkter/MultiView-EEG-AD.git
cd MultiView-EEG-AD

conda create -n multiview-eeg-ad python=3.10 -y
conda activate multiview-eeg-ad
pip install -r requirements.txt
```

Install a PyTorch build compatible with your CUDA version if GPU acceleration
is required. The experiments were designed for Python and PyTorch; no dataset,
pretrained weight, or generated output is included in this repository.

## Data Preparation

Obtain CAUEEG and OpenNeuro `ds004504` from their official sources and comply
with the corresponding licenses and ethics requirements. The preprocessing
pipeline harmonizes the recordings to 19 channels and 128 Hz.

Example commands:

```bash
python scripts/preprocess_caueeg.py \
  --input-dir /path/to/caueeg \
  --output-dir /path/to/processed_caueeg \
  --window-seconds 1 \
  --stride-seconds 1 \
  --target-sfreq 128

python scripts/preprocess_ds004504.py \
  --bids-root /path/to/ds004504 \
  --output-dir /path/to/processed_ds004504 \
  --window-sec 1 \
  --target-sfreq 128
```

Each processed dataset directory should contain:

```text
x_19ch.npy          # [windows, 19, 128]
y_19ch.npy          # binary window labels
sub_ids_19ch.npy    # participant identifier for every window
```

Participant identifiers are required: windows from the same participant must
never be split across training, validation, and test sets.

## Self-Supervised Pretraining

The default pretraining weights match the manuscript objective:

```text
L_pre = L_signal + 0.5 L_spectral + 0.1 L_branch + 0.1 L_temporal-view
```

Run pretraining with hybrid masking:

```bash
python scripts/pretrain_tage_timefreq_consistency.py \
  --data-dir /path/to/unlabeled_processed_eeg \
  --output-dir outputs/pretraining \
  --checkpoint-out checkpoints/multiview_eeg_ad.pth \
  --mask-mode hybrid \
  --mask-ratio 0.3 \
  --signal-weight 1.0 \
  --freq-weight 0.5 \
  --consistency-weight 0.1 \
  --crop-consistency-weight 0.1 \
  --seed 2026
```

## Participant-Level Fine-Tuning

```bash
python scripts/run_dual_stream_subject_split.py \
  --data-dir /path/to/processed_target_eeg \
  --checkpoint checkpoints/multiview_eeg_ad.pth \
  --checkpoint-scope encoders \
  --model-type dual_stream \
  --output-dir outputs/finetuning \
  --alignment euclidean \
  --tune-threshold \
  --seed 2026
```

For a matched random-initialization baseline, replace the checkpoint arguments
with `--no-checkpoint`. Use saved split files and identical seeds for paired
comparisons.

The fine-tuning runner saves the resolved arguments, training history, best
model, summary metrics, and participant-level predictions in the selected
output directory.

## Reproducibility and Leakage Control

- Fix the random seed for every run.
- Split by participant identifier, never by EEG window.
- Estimate Euclidean alignment from training participants only.
- Select classification thresholds using validation participants only.
- Report participant-level Accuracy, Balanced Accuracy, F1, AUROC,
  Sensitivity, and Specificity.
- Keep data, checkpoints, and generated outputs outside version control.

## Data and Weights

Datasets, pretrained checkpoints, and experiment outputs are intentionally not
distributed in this repository. The `.gitignore` excludes common EEG formats,
NumPy arrays, model weights, checkpoints, and output directories.

## Citation

The manuscript is currently under review. A complete BibTeX entry and paper
link will be added after publication. Until then, please cite the repository by
its title and author.

## Acknowledgements

This research code was developed from the Apache-2.0-licensed
[BioFoundation](https://github.com/pulp-bio/BioFoundation) codebase. We thank
its authors and the maintainers of the public EEG datasets used in this study.

## License

This repository is released under the [Apache License 2.0](LICENSE). Dataset
licenses and access conditions remain governed by their respective providers.
