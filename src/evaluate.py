"""
src/evaluate.py — PESQ / STOI / SI-SDR evaluation, compares trained model vs classical baseline.

Run with: python -m src.evaluate
Configure paths/checkpoint via configs/config.yaml (same file src/train.py uses).

Unlike training, evaluation runs on FULL-LENGTH test utterances (not fixed
2-second segments) because PESQ and STOI are defined per-utterance and
truncating changes the score. This intentionally does NOT use
NoisyCleanDataset (which crops/pads for batching) — evaluation processes
one file at a time instead.
"""

import os
import csv

import torch
import yaml
from tqdm import tqdm

from src.dsp import load_audio, stft, istft, magnitude_phase, reconstruct_complex, SAMPLE_RATE
from src.model import UNetSE
from src.baseline import spectral_subtraction

try:
    from pesq import pesq
    from pystoi import stoi
except ImportError as e:
    raise ImportError(
        "pesq and pystoi are required for evaluation. Install with: "
        "pip install pesq pystoi"
    ) from e


def load_config(path="configs/config.yaml"):
    if os.path.exists(path):
        with open(path) as f:
            return yaml.safe_load(f)
    print(f"[evaluate] WARNING: {path} not found, using built-in defaults.")
    return {}


def si_sdr(pred_wav, target_wav, eps=1e-8):
    """Scale-Invariant SDR in dB (higher is better) — the metric, not the loss."""
    pred_wav = pred_wav - pred_wav.mean()
    target_wav = target_wav - target_wav.mean()

    dot = torch.sum(pred_wav * target_wav)
    target_energy = torch.sum(target_wav ** 2) + eps
    s_target = (dot / target_energy) * target_wav
    e_noise = pred_wav - s_target

    return (10 * torch.log10((torch.sum(s_target ** 2) + eps) / (torch.sum(e_noise ** 2) + eps))).item()


def evaluate_pair(clean_wav, enhanced_wav, sr=SAMPLE_RATE):
    """Compute PESQ (wideband), STOI, and SI-SDR for one enhanced/clean pair.

    clean_wav, enhanced_wav: 1D torch tensors, same sample rate.
    """
    min_len = min(clean_wav.shape[-1], enhanced_wav.shape[-1])
    clean_wav = clean_wav[:min_len]
    enhanced_wav = enhanced_wav[:min_len]

    clean_np = clean_wav.numpy()
    enhanced_np = enhanced_wav.numpy()

    try:
        pesq_score = pesq(sr, clean_np, enhanced_np, "wb")
    except Exception as e:
        # pesq raises on near-silent/degenerate clips rather than returning NaN
        print(f"[evaluate] PESQ failed on a clip ({e}), recording as None")
        pesq_score = None

    stoi_score = stoi(clean_np, enhanced_np, sr, extended=False)
    sisdr_score = si_sdr(enhanced_wav, clean_wav)

    return {"PESQ": pesq_score, "STOI": stoi_score, "SI-SDR": sisdr_score}


def enhance_with_model(model, noisy_wav, device):
    spec = stft(noisy_wav.to(device))
    mag, phase = magnitude_phase(spec)
    mag_log = torch.log1p(mag).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        mask = model(mag_log).squeeze()

    enhanced_mag = mask * mag
    enhanced_spec = reconstruct_complex(enhanced_mag, phase)
    return istft(enhanced_spec, length=noisy_wav.shape[-1]).cpu()


def main():
    cfg = load_config()
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("train", {})

    noisy_test_dir = data_cfg.get("noisy_test_dir", "data/raw/noisy_testset_wav")
    clean_test_dir = data_cfg.get("clean_test_dir", "data/raw/clean_testset_wav")
    checkpoint_path = train_cfg.get("checkpoint_path", "checkpoints/unet_se.pt")
    base_ch = train_cfg.get("base_ch", 32)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = None
    if os.path.exists(checkpoint_path):
        model = UNetSE(base_ch=base_ch).to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
        print(f"[evaluate] loaded checkpoint: {checkpoint_path}")
    else:
        print(f"[evaluate] WARNING: no checkpoint at {checkpoint_path} — "
              f"skipping model evaluation, baseline only.")

    clean_files = sorted(os.listdir(clean_test_dir))
    noisy_files = set(os.listdir(noisy_test_dir))
    files = [f for f in clean_files if f in noisy_files]
    print(f"[evaluate] evaluating on {len(files)} test utterances")

    rows = []
    for fname in tqdm(files, desc="evaluate"):
        clean_wav = load_audio(os.path.join(clean_test_dir, fname))
        noisy_wav = load_audio(os.path.join(noisy_test_dir, fname))

        baseline_wav = spectral_subtraction(noisy_wav)
        baseline_metrics = evaluate_pair(clean_wav, baseline_wav)

        row = {"filename": fname,
               "baseline_pesq": baseline_metrics["PESQ"],
               "baseline_stoi": baseline_metrics["STOI"],
               "baseline_sisdr": baseline_metrics["SI-SDR"]}

        if model is not None:
            model_wav = enhance_with_model(model, noisy_wav, device)
            model_metrics = evaluate_pair(clean_wav, model_wav)
            row.update({"model_pesq": model_metrics["PESQ"],
                        "model_stoi": model_metrics["STOI"],
                        "model_sisdr": model_metrics["SI-SDR"]})

        rows.append(row)

    _print_summary(rows, model is not None)
    _save_csv(rows)


def _print_summary(rows, has_model):
    def avg(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else float("nan")

    print("\n" + "=" * 50)
    print("SUMMARY (mean over test set)")
    print("=" * 50)
    print(f"{'Metric':<10} {'Baseline':>12} {'Model':>12}")
    print(f"{'PESQ':<10} {avg('baseline_pesq'):>12.3f} "
          f"{avg('model_pesq') if has_model else float('nan'):>12.3f}")
    print(f"{'STOI':<10} {avg('baseline_stoi'):>12.3f} "
          f"{avg('model_stoi') if has_model else float('nan'):>12.3f}")
    print(f"{'SI-SDR':<10} {avg('baseline_sisdr'):>12.3f} "
          f"{avg('model_sisdr') if has_model else float('nan'):>12.3f}")


def _save_csv(rows, path="evaluation_results.csv"):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[evaluate] per-file results saved to {path}")


if __name__ == "__main__":
    main()