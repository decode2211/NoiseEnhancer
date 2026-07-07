# Dataset Setup Guide — VoiceBank+DEMAND

This guide explains how to download and set up the dataset used in this project, so anyone
cloning the repo can reproduce the exact same data setup on their own machine.

---

## 1. Prerequisites

- **~6 GB free disk space** (zips + extracted files)
- A terminal with `wget` and `unzip` available:
  - **macOS:** built-in terminal works out of the box (install `wget` via `brew install wget`
    if missing)
  - **Linux:** works out of the box
  - **Windows:** use **WSL** (Windows Subsystem for Linux) or **Git Bash**. If you don't have
    either set up, see the "Windows without WSL" section below for a manual alternative.

---

## 2. Which files to download

The Edinburgh DataShare page (https://datashare.ed.ac.uk/handle/10283/2791) lists several
files. You only need **4 of them**:

| File | Size | Purpose |
|---|---|---|
| `clean_trainset_28spk_wav.zip` | 2.32 GB | Clean training speech |
| `noisy_trainset_28spk_wav.zip` | 2.64 GB | Noisy training speech (paired with above) |
| `clean_testset_wav.zip` | 147 MB | Clean test speech |
| `noisy_testset_wav.zip` | 163 MB | Noisy test speech |

**Skip these** (not needed for speech enhancement training):
- `clean_trainset_56spk_wav.zip` / `noisy_trainset_56spk_wav.zip` — a larger alternative split;
  skip it so your results stay comparable to the majority of published papers, which benchmark
  on the 28-speaker version.
- `trainset_28spk_txt.zip` / `testset_txt.zip` — text transcripts, only needed for ASR-related tasks.
- `logfiles.zip` — metadata logs, not needed.

---

## 3. Automated download (Mac/Linux/WSL/Git Bash)

From your project root (the folder containing `src/`, `README.md`, etc.):

```bash
chmod +x download_data.sh
./download_data.sh
```

This will:
1. Create a `data/raw/` folder
2. Download the 4 required zip files (resumable if your connection drops — it uses `wget -c`)
3. Extract them automatically
4. Print the resulting folder structure

Expected result:
data/raw/
├── clean_trainset_28spk_wav/
├── noisy_trainset_28spk_wav/
├── clean_testset_wav/
└── noisy_testset_wav/

---

## 4. Manual download (if you prefer using a browser, or you're on Windows without WSL)

1. Go to https://datashare.ed.ac.uk/handle/10283/2791
2. Click to download each of the 4 files listed in Section 2
3. Create a folder called `data/raw/` inside your project directory
4. Move all 4 downloaded `.zip` files into `data/raw/`
5. Extract each zip **in place** (right-click → Extract Here, or double-click on macOS)
6. Delete the `.zip` files afterward if you want to save disk space (optional)

You should end up with the same folder structure shown in Section 3.

---

## 5. Verifying the download

Run this quick check to make sure everything downloaded and extracted correctly:

```bash
python verify_data.py
```

(See `verify_data.py` — checks file counts and plays/plots a sample pair.)

Expected counts:
- `clean_trainset_28spk_wav/`: **11,572** files
- `noisy_trainset_28spk_wav/`: **11,572** files
- `clean_testset_wav/`: **824** files
- `noisy_testset_wav/`: **824** files

If your counts don't match, the zip extraction was likely incomplete — re-download and
re-extract the mismatched folder.

---

## 6. Notes

- All audio is at **48kHz** as provided — your data pipeline (`src/dsp.py`) resamples to 16kHz
  on load, matching what most published speech enhancement papers use.
- Filenames match exactly between the clean and noisy folders (e.g., `p226_001.wav` appears in
  both `clean_trainset_28spk_wav/` and `noisy_trainset_28spk_wav/`), which is what
  `NoisyCleanDataset` in `src/data_loader.py` relies on for pairing.
- Do not shuffle or rename files across folders — the pairing depends on matching filenames.

