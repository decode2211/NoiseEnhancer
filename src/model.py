"""
src/model.py — U-Net that predicts a Time-Frequency mask.

Input:  noisy log-magnitude spectrogram, shape (B, 1, F, T)
Output: mask in [0, 1], same shape (B, 1, F, T), applied to noisy magnitude
        to get the estimated clean magnitude (see train.py / inference.py).

Architecture note: with N_FFT=512 (dsp.py), F = N_FFT/2 + 1 = 257 frequency
bins. 257 is odd, and this U-Net downsamples F by 2x three times (MaxPool2d).
257 -> 128 -> 64 -> 32 loses 1 pixel at each pooling step due to integer
division, so the encoder/decoder skip connections can end up with mismatched
spatial sizes. `forward()` below handles this with center-cropping/padding
so the network still runs correctly regardless of odd input sizes — but if
you change N_FFT, prefer picking a value where the frequency/time dims are
divisible by 8 (2^3, for 3 pooling stages) to avoid the crop path entirely.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


def _match_size(x, target):
    """Center-crop or pad x (B,C,H,W) so its spatial dims match target's."""
    _, _, h, w = x.shape
    _, _, th, tw = target.shape

    if h != th or w != tw:
        dh, dw = th - h, tw - w
        x = F.pad(x, (0, max(dw, 0), 0, max(dh, 0)))
        if dh < 0 or dw < 0:
            x = x[:, :, :th, :tw]
    return x


class UNetSE(nn.Module):
    """U-Net that predicts a Time-Frequency mask (values 0-1) applied to noisy magnitude."""

    def __init__(self, base_ch=32):
        super().__init__()
        self.enc1 = conv_block(1, base_ch)
        self.enc2 = conv_block(base_ch, base_ch * 2)
        self.enc3 = conv_block(base_ch * 2, base_ch * 4)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = conv_block(base_ch * 4, base_ch * 8)

        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 2, stride=2)
        self.dec3 = conv_block(base_ch * 8, base_ch * 4)
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec2 = conv_block(base_ch * 4, base_ch * 2)
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.dec1 = conv_block(base_ch * 2, base_ch)

        self.out_conv = nn.Conv2d(base_ch, 1, 1)

    def forward(self, x):
        # x: (batch, 1, freq, time)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        d3 = self.up3(b)
        d3 = _match_size(d3, e3)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = _match_size(d2, e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = _match_size(d1, e1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        d1 = _match_size(d1, x)  # guarantee output exactly matches input shape
        mask = torch.sigmoid(self.out_conv(d1))  # values in [0,1]
        return mask


# ---------------------------------------------------------------------------
# Smoke test — run `python -m src.model`
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model = UNetSE(base_ch=32)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"UNetSE parameter count: {n_params:,}")

    # Simulate a batch: (B=2, C=1, F=257, T=126) — matches N_FFT=512 output
    # for a 2-second clip at 16kHz with hop_length=128.
    dummy = torch.randn(2, 1, 257, 126)
    mask = model(dummy)

    print(f"Input shape:  {tuple(dummy.shape)}")
    print(f"Output shape: {tuple(mask.shape)}")
    assert mask.shape == dummy.shape, "Output shape must match input shape!"
    assert mask.min() >= 0 and mask.max() <= 1, "Mask values must be in [0, 1]!"
    print("✅ Forward pass OK, output shape matches input, mask range is valid.")