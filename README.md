# Speech Enhancement / Noise Suppression using AI + Signal Processing

A deep learning based speech enhancement system that removes background noise from speech
recordings, combining classical Digital Signal Processing (STFT/ISTFT, spectral masking) with
a neural network (U-Net style) trained on the VoiceBank+DEMAND dataset.

> See `theory.md` for the full conceptual/mathematical explanation of every step.

---

## 1. Overview

**Input:** Noisy speech waveform (speech + background noise)
**Output:** Enhanced/denoised speech waveform

**Pipeline:**
```
Noisy waveform → STFT → Magnitude + Phase split → Neural Network (predicts mask)
→ Mask x Noisy Magnitude → Combine with noisy Phase → ISTFT → Enhanced waveform
```

---

## 2. Dataset

**VoiceBank+DEMAND** — standard benchmark dataset for speech enhancement.
- Clean speech: VCTK corpus recordings
- Noise: DEMAND dataset (10 real-world noise types) mixed at multiple SNR levels
- Already provided as paired (noisy, clean) `.wav` files, split into train/test

Download link and directory structure expected:
```
data/raw/
├── clean_trainset_wav/
├── noisy_trainset_wav/
├── clean_testset_wav/
└── noisy_testset_wav/
```

---

## 3. Repository Structure

```
speech-enhancement/
│
├── data/
│   ├── raw/                # original VoiceBank+DEMAND files
│   └── processed/          # precomputed STFT features (cached .npy/.pt files)
│
├── src/
│   ├── dsp.py               # STFT/ISTFT, feature extraction utilities
│   ├── data_loader.py       # PyTorch Dataset + DataLoader
│   ├── model.py             # U-Net architecture
│   ├── losses.py            # loss functions (MSE, log-MSE, SI-SDR)
│   ├── train.py              # training loop
│   ├── evaluate.py           # PESQ / STOI / SI-SDR computation
│   ├── baseline.py            # classical spectral subtraction / Wiener filter baseline
│   └── inference.py           # run trained model on a new noisy file
│
├── notebooks/
│   └── exploration.ipynb     # visualize spectrograms, sanity-check STFT/ISTFT
│
├── demo/
│   └── app.py                 # Streamlit/Gradio demo app
│
├── checkpoints/                # saved model weights (.pt files)
├── configs/
│   └── config.yaml             # hyperparameters and paths
├── requirements.txt
└── README.md
```

---

## 4. Setup

```bash
git clone <your-repo-url>
cd speech-enhancement
python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**requirements.txt**
```
torch
torchaudio
librosa
numpy
scipy
pesq
pystoi
matplotlib
streamlit
soundfile
pyyaml
tqdm
```

---

## 5. Core Code

### 5.1 `src/dsp.py` — STFT / ISTFT utilities

```python
import torch
import torchaudio

SAMPLE_RATE = 16000
N_FFT = 512          # ~32ms window at 16kHz
HOP_LENGTH = 128      # ~8ms hop (75% overlap)
WIN_LENGTH = 512

window = torch.hann_window(WIN_LENGTH)

def load_audio(path, sr=SAMPLE_RATE):
    wav, orig_sr = torchaudio.load(path)
    if orig_sr != sr:
        wav = torchaudio.functional.resample(wav, orig_sr, sr)
    return wav.squeeze(0)  # mono

def stft(wav):
    return torch.stft(
        wav, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
        window=window, return_complex=True
    )

def istft(spec, length=None):
    return torch.istft(
        spec, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
        window=window, length=length
    )

def magnitude_phase(spec):
    mag = torch.abs(spec)
    phase = torch.angle(spec)
    return mag, phase

def reconstruct_complex(mag, phase):
    return mag * torch.exp(1j * phase)
```

**Sanity check (always do this first):** load a clean wav → STFT → ISTFT → confirm the
reconstructed audio is (near) identical to the original. If this fails, everything downstream
will be wrong.

```python
wav = load_audio("sample_clean.wav")
spec = stft(wav)
recon = istft(spec, length=wav.shape[-1])
print(torch.allclose(wav, recon, atol=1e-4))   # should be True
```

---

### 5.2 `src/data_loader.py` — Dataset class

```python
import os
import torch
from torch.utils.data import Dataset
from src.dsp import load_audio, stft, magnitude_phase

class NoisyCleanDataset(Dataset):
    def __init__(self, noisy_dir, clean_dir):
        self.noisy_files = sorted(os.listdir(noisy_dir))
        self.noisy_dir = noisy_dir
        self.clean_dir = clean_dir

    def __len__(self):
        return len(self.noisy_files)

    def __getitem__(self, idx):
        fname = self.noisy_files[idx]
        noisy_wav = load_audio(os.path.join(self.noisy_dir, fname))
        clean_wav = load_audio(os.path.join(self.clean_dir, fname))

        min_len = min(noisy_wav.shape[-1], clean_wav.shape[-1])
        noisy_wav, clean_wav = noisy_wav[:min_len], clean_wav[:min_len]

        noisy_spec = stft(noisy_wav)
        clean_spec = stft(clean_wav)

        noisy_mag, noisy_phase = magnitude_phase(noisy_spec)
        clean_mag, _ = magnitude_phase(clean_spec)

        # log-magnitude, commonly used to compress dynamic range
        noisy_mag_log = torch.log1p(noisy_mag)
        clean_mag_log = torch.log1p(clean_mag)

        return noisy_mag_log, clean_mag_log, noisy_phase
```

---

### 5.3 `src/model.py` — U-Net for mask prediction

```python
import torch
import torch.nn as nn

def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )

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
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        mask = torch.sigmoid(self.out_conv(d1))  # values in [0,1]
        return mask
```

---

### 5.4 `src/losses.py` — Loss functions

```python
import torch
import torch.nn as nn

mse_loss = nn.MSELoss()

def log_magnitude_loss(pred_log_mag, target_log_mag):
    return mse_loss(pred_log_mag, target_log_mag)

def si_sdr_loss(pred_wav, target_wav, eps=1e-8):
    """Scale-Invariant SDR loss (negative, since we minimize)."""
    pred_wav = pred_wav - pred_wav.mean(dim=-1, keepdim=True)
    target_wav = target_wav - target_wav.mean(dim=-1, keepdim=True)

    s_target = (torch.sum(pred_wav * target_wav, dim=-1, keepdim=True) /
                (torch.sum(target_wav ** 2, dim=-1, keepdim=True) + eps)) * target_wav
    e_noise = pred_wav - s_target

    si_sdr = 10 * torch.log10(
        (torch.sum(s_target ** 2, dim=-1) + eps) /
        (torch.sum(e_noise ** 2, dim=-1) + eps)
    )
    return -si_sdr.mean()
```

---

### 5.5 `src/train.py` — Training loop (skeleton)

```python
import torch
from torch.utils.data import DataLoader
from src.data_loader import NoisyCleanDataset
from src.model import UNetSE
from src.losses import log_magnitude_loss

device = "cuda" if torch.cuda.is_available() else "cpu"

train_dataset = NoisyCleanDataset("data/raw/noisy_trainset_wav", "data/raw/clean_trainset_wav")
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=None)

model = UNetSE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 30

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for noisy_mag, clean_mag, noisy_phase in train_loader:
        noisy_mag = noisy_mag.unsqueeze(1).to(device)   # (B,1,F,T)
        clean_mag = clean_mag.unsqueeze(1).to(device)

        mask = model(noisy_mag)
        pred_mag = mask * noisy_mag

        loss = log_magnitude_loss(pred_mag, clean_mag)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {running_loss/len(train_loader):.4f}")

torch.save(model.state_dict(), "checkpoints/unet_se.pt")
```

> Note: real audio clips have variable length — in practice you'll want to chunk/pad
> spectrograms to a fixed number of time frames per batch (see `theory.md` §7 for details).

---

### 5.6 `src/evaluate.py` — PESQ / STOI / SI-SDR evaluation

```python
from pesq import pesq
from pystoi import stoi
import numpy as np

def evaluate_pair(clean_wav, enhanced_wav, sr=16000):
    clean_np = clean_wav.numpy()
    enhanced_np = enhanced_wav.numpy()

    pesq_score = pesq(sr, clean_np, enhanced_np, 'wb')   # wideband PESQ
    stoi_score = stoi(clean_np, enhanced_np, sr, extended=False)

    return {"PESQ": pesq_score, "STOI": stoi_score}
```

---

### 5.7 `src/baseline.py` — Classical baseline (Spectral Subtraction)

```python
import torch
from src.dsp import stft, istft, magnitude_phase, reconstruct_complex

def spectral_subtraction(noisy_wav, noise_estimate_frames=6, alpha=2.0, beta=0.01):
    """Simple spectral subtraction baseline for comparison against the AI model."""
    spec = stft(noisy_wav)
    mag, phase = magnitude_phase(spec)

    # estimate noise spectrum from the first few frames (assumed noise-only)
    noise_mag = mag[:, :noise_estimate_frames].mean(dim=1, keepdim=True)

    enhanced_mag = mag - alpha * noise_mag
    enhanced_mag = torch.clamp(enhanced_mag, min=beta * mag)  # spectral floor

    enhanced_spec = reconstruct_complex(enhanced_mag, phase)
    return istft(enhanced_spec, length=noisy_wav.shape[-1])
```

---

### 5.8 `src/inference.py` — Run enhancement on a new file

```python
import torch
import torchaudio
from src.dsp import load_audio, stft, istft, magnitude_phase, reconstruct_complex
from src.model import UNetSE

def enhance_file(input_path, output_path, checkpoint="checkpoints/unet_se.pt"):
    model = UNetSE()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()

    wav = load_audio(input_path)
    spec = stft(wav)
    mag, phase = magnitude_phase(spec)
    mag_log = torch.log1p(mag).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        mask = model(mag_log).squeeze()

    enhanced_mag = mask * mag
    enhanced_spec = reconstruct_complex(enhanced_mag, phase)
    enhanced_wav = istft(enhanced_spec, length=wav.shape[-1])

    torchaudio.save(output_path, enhanced_wav.unsqueeze(0), 16000)
```

---

### 5.9 `demo/app.py` — Streamlit demo (skeleton)

```python
import streamlit as st
import tempfile
from src.inference import enhance_file

st.title("AI Speech Enhancement Demo")

uploaded_file = st.file_uploader("Upload a noisy audio file (.wav)", type=["wav"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp_in:
        tmp_in.write(uploaded_file.read())
        tmp_in.flush()

        st.subheader("Noisy Input")
        st.audio(tmp_in.name)

        output_path = "enhanced_output.wav"
        enhance_file(tmp_in.name, output_path)

        st.subheader("Enhanced Output")
        st.audio(output_path)
```

Run with:
```bash
streamlit run demo/app.py
```

---

## 6. Roadmap / Milestones

1. [ ] Verify STFT → ISTFT reconstruction is lossless
2. [ ] Implement and test classical baseline (spectral subtraction)
3. [ ] Build data pipeline + sanity check on a few samples
4. [ ] Train U-Net mask model, confirm loss decreases
5. [ ] Evaluate with PESQ / STOI / SI-SDR, compare to baseline
6. [ ] Tune loss function (try log-mag loss vs SI-SDR loss)
7. [ ] Build Streamlit demo
8. [ ] (Stretch) Implement Conv-TasNet and compare time-domain vs T-F domain approaches
9. [ ] Write up final report with metric tables + before/after spectrogram plots

---

## 7. References

- Pascual et al., "SEGAN: Speech Enhancement Generative Adversarial Network"
- Luo & Mesgarani, "Conv-TasNet: Surpassing Ideal Time-Frequency Magnitude Masking for Speech Separation"
- Valentini-Botinhao et al., "Noisy speech database for training speech enhancement algorithms" (VoiceBank+DEMAND)
- See `theory.md` for full conceptual background.