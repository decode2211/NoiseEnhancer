"""
Quick sanity check that the VoiceBank+DEMAND dataset downloaded/extracted correctly.

Usage:
    python verify_data.py
"""

import os

RAW_DIR = "data/raw"

EXPECTED_COUNTS = {
    "clean_trainset_28spk_wav": 11572,
    "noisy_trainset_28spk_wav": 11572,
    "clean_testset_wav": 824,
    "noisy_testset_wav": 824,
}


def check_folder(name, expected_count):
    path = os.path.join(RAW_DIR, name)
    if not os.path.isdir(path):
        print(f"[MISSING] {path} does not exist.")
        return False

    files = [f for f in os.listdir(path) if f.endswith(".wav")]
    count = len(files)
    status = "OK" if count == expected_count else "MISMATCH"
    print(f"[{status}] {name}: found {count} files (expected {expected_count})")
    return count == expected_count


def check_pairing():
    """Confirm every noisy train file has a matching clean train file (same filename)."""
    clean_dir = os.path.join(RAW_DIR, "clean_trainset_28spk_wav")
    noisy_dir = os.path.join(RAW_DIR, "noisy_trainset_28spk_wav")

    if not (os.path.isdir(clean_dir) and os.path.isdir(noisy_dir)):
        print("[SKIP] Cannot check pairing, train folders missing.")
        return

    clean_files = set(os.listdir(clean_dir))
    noisy_files = set(os.listdir(noisy_dir))

    missing_in_noisy = clean_files - noisy_files
    missing_in_clean = noisy_files - clean_files

    if not missing_in_noisy and not missing_in_clean:
        print("[OK] All clean/noisy training filenames match perfectly.")
    else:
        print(f"[MISMATCH] {len(missing_in_noisy)} files in clean but not noisy, "
              f"{len(missing_in_clean)} files in noisy but not clean.")


def play_sample():
    """Optional: load one clean/noisy pair and report basic info (requires torchaudio)."""
    try:
        import torchaudio
    except ImportError:
        print("[SKIP] torchaudio not installed, skipping audio load check.")
        return

    clean_dir = os.path.join(RAW_DIR, "clean_trainset_28spk_wav")
    noisy_dir = os.path.join(RAW_DIR, "noisy_trainset_28spk_wav")

    if not os.path.isdir(clean_dir):
        return

    sample_files = sorted(os.listdir(clean_dir))
    if not sample_files:
        return

    sample_name = sample_files[0]
    clean_wav, sr_c = torchaudio.load(os.path.join(clean_dir, sample_name))
    noisy_path = os.path.join(noisy_dir, sample_name)

    if os.path.exists(noisy_path):
        noisy_wav, sr_n = torchaudio.load(noisy_path)
        print(f"[OK] Loaded sample pair '{sample_name}': "
              f"clean shape={tuple(clean_wav.shape)} sr={sr_c}, "
              f"noisy shape={tuple(noisy_wav.shape)} sr={sr_n}")
    else:
        print(f"[MISMATCH] '{sample_name}' exists in clean set but not in noisy set.")


if __name__ == "__main__":
    print("Checking dataset folders...\n")
    all_ok = True
    for folder, expected in EXPECTED_COUNTS.items():
        ok = check_folder(folder, expected)
        all_ok = all_ok and ok

    print()
    check_pairing()

    print()
    play_sample()

    print("\nDone." if all_ok else "\nSome checks failed — see messages above.")