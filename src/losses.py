"""
src/losses.py — Loss functions for training the mask-prediction U-Net.

Two options, matching README milestone 6 ("tune loss function (try log-mag
loss vs SI-SDR loss)"):

1. log_magnitude_loss — simple MSE in the log-magnitude spectrogram domain.
   Fast, stable, good default to confirm the model can learn at all.

2. si_sdr_loss — Scale-Invariant SDR, computed on the time-domain waveform.
   Correlates better with perceptual quality (and with PESQ/STOI), but
   requires reconstructing the waveform (mask -> magnitude -> ISTFT) inside
   the training loop, which is slower and needs the noisy phase.
"""

import torch
import torch.nn as nn

mse_loss = nn.MSELoss()


def log_magnitude_loss(pred_log_mag, target_log_mag):
    """MSE between predicted and target log-magnitude spectrograms."""
    return mse_loss(pred_log_mag, target_log_mag)


def si_sdr_loss(pred_wav, target_wav, eps=1e-8):
    """
    Scale-Invariant SDR loss (negative, since we minimize).

    pred_wav, target_wav: (B, T) time-domain waveforms.
    """
    # Zero-mean both signals first — SI-SDR is defined on zero-mean signals.
    pred_wav = pred_wav - pred_wav.mean(dim=-1, keepdim=True)
    target_wav = target_wav - target_wav.mean(dim=-1, keepdim=True)

    # Project pred onto target direction (the "signal" component).
    dot = torch.sum(pred_wav * target_wav, dim=-1, keepdim=True)
    target_energy = torch.sum(target_wav ** 2, dim=-1, keepdim=True) + eps
    s_target = (dot / target_energy) * target_wav

    e_noise = pred_wav - s_target

    si_sdr = 10 * torch.log10(
        (torch.sum(s_target ** 2, dim=-1) + eps) /
        (torch.sum(e_noise ** 2, dim=-1) + eps)
    )
    return -si_sdr.mean()


def combined_loss(pred_log_mag, target_log_mag, pred_wav=None, target_wav=None,
                   si_sdr_weight=0.0):
    """
    Convenience wrapper: log-mag loss, optionally blended with SI-SDR loss
    if waveforms are provided and si_sdr_weight > 0. Lets you try both
    losses (or a mix) without changing train.py's call site.
    """
    loss = log_magnitude_loss(pred_log_mag, target_log_mag)
    if si_sdr_weight > 0:
        if pred_wav is None or target_wav is None:
            raise ValueError("si_sdr_weight > 0 requires pred_wav and target_wav")
        loss = loss + si_sdr_weight * si_sdr_loss(pred_wav, target_wav)
    return loss


# ---------------------------------------------------------------------------
# Smoke test — run `python -m src.losses`
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    # log-magnitude loss: identical inputs -> 0 loss
    a = torch.randn(2, 257, 126)
    loss_same = log_magnitude_loss(a, a)
    print(f"log_magnitude_loss(a, a) = {loss_same.item():.6f} (should be ~0)")

    b = a + torch.randn_like(a) * 0.5
    loss_diff = log_magnitude_loss(a, b)
    print(f"log_magnitude_loss(a, b) = {loss_diff.item():.6f} (should be > 0)")
    assert loss_diff.item() > loss_same.item()

    # si_sdr_loss: identical waveforms -> very negative loss (high SI-SDR)
    wav = torch.randn(2, 16000)
    si_sdr_same = si_sdr_loss(wav, wav)
    print(f"si_sdr_loss(wav, wav)     = {si_sdr_same.item():.2f} dB (very negative = perfect match)")

    noisy_wav = wav + torch.randn_like(wav) * 0.3
    si_sdr_diff = si_sdr_loss(noisy_wav, wav)
    print(f"si_sdr_loss(noisy, wav)   = {si_sdr_diff.item():.2f} dB (should be higher / less negative)")
    assert si_sdr_diff.item() > si_sdr_same.item()

    print("✅ Both loss functions behave correctly (identical inputs minimize loss).")