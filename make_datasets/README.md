Copyright (C) 2025 ETH Zurich, Switzerland. SPDX-License-Identifier: Apache-2.0. See LICENSE file at the root of the repository for details.

# Dataset Preparation Scripts

This directory contains scripts for processing raw datasets into formats suitable for efficient training and evaluation.

## .SET File Conversion (`convert_set_to_hdf5.py`)

This script converts EEGLAB .set files to HDF5 format for use with the BioFoundation project. It's particularly useful when working with custom EEG datasets that are in .set format.

**Key Features**:
- Converts EEGLAB .set files to standardized HDF5 format
- Automatically handles data preprocessing (filtering, resampling, channel selection)
- Splits data into train/validation/test sets (70%/15%/15%)
- Supports both pretraining and finetuning modes
- Handles windowing of continuous EEG data

**Prerequisites**:
- Raw .set files from EEGLAB
- Required Python packages: `mne`, `numpy`, `scipy`, `h5py`, `tqdm`
- Output directory structure with train/val/test subdirectories

**Directory Structure Requirements**:
```
output_directory/
├── train/          # Will contain train.h5
├── val/            # Will contain val.h5  
├── test/           # Will contain test.h5
└── temp_pickle/    # Temporary directory (automatically created and cleaned)
```

**Command-Line Usage**:
```bash
python make_datasets/convert_set_to_hdf5.py --set_dir /path/to/set/files --output_dir /path/to/output/directory [options]
```

**Required Arguments**:
- `--set_dir`: Directory containing .set files to convert
- `--output_dir`: Output directory (must contain train/val/test subdirectories)

**Optional Arguments**:
- `--window_duration`: Window duration in seconds (default: 5)
- `--label`: Label to assign to all samples (optional)
- `--task_type`: Task type - "pretrain" or "finetune" (default: finetune)

**Processing Pipeline**:
1. **Read .set files**: Load EEGLAB format files using scipy.io.loadmat
2. **Preprocess data**: 
   - Standardize channel ordering (FP1, FP2, F3, F4, etc.)
   - Apply band-pass filter (0.1-75 Hz) and 60 Hz notch filter
   - Resample to 256 Hz
3. **Window data**: Split continuous recordings into fixed-size windows
4. **Split data**: Distribute windows into train (70%), validation (15%), test (15%) sets
5. **Create HDF5**: Convert pickle files to efficient HDF5 format
6. **Cleanup**: Remove temporary files

**Examples**:

Basic usage:
```bash
# Create required directory structure first
mkdir -p ./processed_data/{train,val,test}

# Convert .set files
python make_datasets/convert_set_to_hdf5.py --set_dir ./raw_set_files --output_dir ./processed_data
```

With custom parameters:
```bash
python make_datasets/convert_set_to_hdf5.py \
    --set_dir ./eeg_data/set_files \
    --output_dir ./model_data \
    --window_duration 10 \
    --label 1 \
    --task_type finetune
```

Pretraining mode (no labels):
```bash
python make_datasets/convert_set_to_hdf5.py \
    --set_dir ./eeg_data/set_files \
    --output_dir ./model_data \
    --task_type pretrain
```

**Output Format**:
The script generates three HDF5 files with the following structure:
```
train.h5/
├── data_group_0/
│   ├── X: [n_samples, n_channels, window_length] - EEG data
│   └── y: [n_samples] - Labels (only in finetune mode)
├── data_group_1/
│   ├── X: ...
│   └── y: ...
└── ...

val.h5/ (same structure)
test.h5/ (same structure)
```

**Troubleshooting**:
- **ImportError for h5py**: Install with `pip install h5py`
- **No .set files found**: Check that --set_dir contains .set files
- **Missing output directories**: Create train/val/test subdirectories in output_dir
- **Memory issues**: Reduce window_duration or process fewer files at once

---

## TUEG preparation (`make_tueg_bipolar.py`)

We provide an example script on how to format TUEG into bipolar. This is then used by our pre-training scripts. Make sure to change all of the occurences of `#CHANGEME`. The main processing script follow a very similar pattern to the one explained for `process_raw_eeg.py` here below.

**Prerequisites**:
-   You must have the raw TUEG dataset downloaded from their official sources.

**Command-Line Usage**:
The script takes the dataset name, the path to the raw data, and an output directory as arguments.

```bash
python make_datasets/make_tueg_bipolar.py --in_path /path/to/raw_data --hdf5_file_path /path/to/processed_data
```

### Arguments:
-   `--in_path`: The absolute path to the root directory of the downloaded raw EDF files for the TUEG dataset.
-   `--hdf5_file_path`: The directory where the script will save the procsessed `.h5` files.

---

## Raw EEG Preprocessing (`process_raw_eeg.py`)

Before the datasets can be converted to HDF5, the raw EDF files must first be preprocessed, windowed, and saved as intermediate
pickle files. The `process_raw_eeg.py` script handles this entire pipeline.

**Key Processing Steps**
1. **Loads EDF files**: Reads the raw EEG data.
2. **Selects and orders channels**: Standardizes all recordings to a 21‑channel layout.
3. **Applies filters**: A band‑pass filter (0.1–75.0 Hz) and a 60 Hz notch filter are applied.
4. **Resamples data**: All data is resampled to 256 Hz.
5. **Creates bipolar montage**: Re‑references the signals to the standard TCP bipolar montage.
6. **Windows data**: Segments the continuous signal into 5‑second windows.
7. **Generates labels**: Applies labels based on the dataset and specified mode:
   - **tuab**: Assigns a single file‑level label (normal/abnormal) to all windows from that file.
    - **tusl**: Assigns a segment‑level label (`bckg`, `seiz`, `slow`) based on the majority label found in the 5‑second window.
    - **tuar**: Assigns segment‑level artifact labels based on the chosen `--mode` (e.g., a single 0/1 for **Binary** or a per‑channel array for **MultiLabel**).
8. **Saves segments**: Each 5‑second window (data + label) is saved as a separate `.pkl` file.

**How to Use the Script**
The script is designed to be run from the command line and requires you to specify which dataset you are processing.

**Prerequisites**
- You must have the raw **TUH** datasets (**TUAB**, **TUSL**, or **TUAR**) downloaded from their official sources.
- The required Python packages, including `mne`, `numpy`, `pandas`, and `tqdm`, must be installed.

**Command‑Line Usage**
```bash
python make_datasets/process_raw_eeg.py <dataset_name>       --root_dir /path/to/raw_data       --output_dir /processed_eeg [options]
```

**Arguments**
- `<dataset_name>`: The dataset to process. Must be one of `tuab`, `tusl`, or `tuar`.
- `--root_dir`: The absolute path to the root directory of the downloaded raw EDF files for the specified dataset.
- `--output_dir`: The directory where the script will save the processed `.pkl` files. The script will automatically create `train`, `val`, and `test` subdirectories inside a `processed` folder here.
- `--processes`: (Optional) The number of CPU cores to use for parallel processing (default: `24`).
- `--mode`: (Optional) Specifies the labeling mode for the **TUAR** dataset only. Must be one of `Binary`, `MultiBinary`, or `MultiLabel` (default: `Binary`).

**Examples**
- To process the **TUAB** dataset:
  ```bash
  python make_datasets/process_raw_eeg.py tuab         --root_dir /eeg_data/TUAB/edf         --output_dir /processed_eeg
  ```

- To process the **TUAR** dataset using the **MultiLabel** mode:
  ```bash
  python make_datasets/process_raw_eeg.py tuar         --root_dir /eeg_data/TUAR/edf         --output_dir /processed_eeg       --mode MultiLabel
  ```

---

## HDF5 Conversion (`make_hdf5.py`)

After you have generated the processed `.pkl` files using the script above, the `make_hdf5.py` script is used to bundle these
thousands of small pickle files into large, efficient HDF5 (`.h5`) files. This is the final step before training.

**How to Use the Script**
The script is run from the command line and takes the path to your processed data as an argument.

**Command‑Line Usage**
```bash
python make_datasets/make_hdf5.py --prepath /processed_eeg [options]
```

**Arguments**
- `--prepath` (Required): The absolute path to the directory containing your dataset folders (e.g., `/processed_eeg/`). This directory
  should contain the `TUAB_data`, `TUSL_data`, etc., folders created by `process_raw_eeg.py`, so this should mirror the `--output_dir` used previously.
- `--dataset` (Optional): Which dataset to process. Choices: `TUAR`, `TUSL`, `TUAB`, `All` (default: `All`).
- `--remove_pkl` (Optional): If included, this flag will delete the `processed` directory (containing all the intermediate `.pkl` files)
  for a dataset after its `.h5` files are successfully created.

**Examples**
- To convert all datasets found in `/processed_eeg` and remove the `.pkl` files after conversion:
  ```bash
  python make_datasets/make_hdf5.py --prepath /processed_eeg --dataset All --remove_pkl
  ```

- To convert only the **TUAB** dataset and keep the original `.pkl` files:
  ```bash
  python make_datasets/make_hdf5.py --prepath /processed_eeg --dataset TUAB
  ```