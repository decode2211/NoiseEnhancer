# Theory: Speech Enhancement using Signal Processing + AI

This document explains the conceptual and mathematical foundations behind every stage of the
speech enhancement pipeline described in `README.md`.

---

## 1. Problem Formulation

A noisy speech signal is modeled as:

```
x(t) = s(t) + n(t)
```

where:
- `x(t)` = observed noisy signal
- `s(t)` = clean speech signal (what we want to recover)
- `n(t)` = additive noise (assumed uncorrelated with speech)

Our goal is to find an estimator `ŝ(t)` that is as close as possible to `s(t)`, given only `x(t)`.

This is fundamentally an **inverse problem**: we observe a mixture and must separate/estimate
one component. Because noise is unpredictable and speech has complex structure, classical
model-based methods (spectral subtraction, Wiener filtering) make simplifying statistical
assumptions, while deep learning methods learn the mapping directly from data.

---

## 2. Why Work in the Time-Frequency Domain?

Speech and most real-world noises have very different structures when viewed in frequency:
- Speech energy is concentrated in specific formant bands and changes over time (non-stationary).
- Many noises are broadband or have different spectral shapes.

Working directly on the raw waveform makes it hard for a model to exploit this structural
difference. Converting to a **time-frequency (T-F) representation** via the **Short-Time
Fourier Transform (STFT)** makes the frequency structure explicit and lets us apply
convolutional/2D models the same way we would for images (spectrograms are just 2D data:
frequency x time).

---

## 3. Short-Time Fourier Transform (STFT)

Speech is non-stationary — its frequency content changes over time — so we can't just take one
FFT of the entire signal. Instead, STFT:

1. **Frames** the signal into short overlapping segments (~20–32ms), since speech is roughly
   stationary within such a short window.
2. **Windows** each frame (commonly a **Hann window**) to reduce spectral leakage caused by
   artificially cutting the signal into segments.
3. **Applies FFT** to each windowed frame to get its frequency content.
4. Frames overlap (commonly 50–75%) so the transform captures smooth transitions and allows
   perfect reconstruction later.

Mathematically, for frame index `m` and window function `w(n)`:

```
X(m, k) = Σ_n  x(n) * w(n - mH) * e^(-j2πkn/N)
```

where `H` is the hop size (step between frames), `N` is the FFT size, and `k` is the frequency
bin index.

The result `X(m, k)` is a **complex-valued matrix**: one axis is time (frame index), the other
is frequency (bin index). Each value has:
- **Magnitude** `|X(m,k)|` — how much energy is present at that time-frequency point
- **Phase** `∠X(m,k)` — the phase offset of that frequency component

### Key parameter choices

| Parameter | Typical value (16kHz speech) | Why |
|---|---|---|
| Window length | 25–32 ms (~512 samples) | Long enough to resolve frequency, short enough that speech is quasi-stationary within it |
| Hop length | 25% of window (~8ms) | Overlap needed for smooth reconstruction (COLA condition) |
| Window type | Hann | Smooth taper at edges, reduces spectral leakage, satisfies perfect-reconstruction (COLA) constraints when combined with appropriate hop size |
| FFT size | ≥ window length (often same or zero-padded) | Determines frequency resolution |

### Trade-off: Time resolution vs Frequency resolution

This is the core STFT trade-off (related to the Heisenberg uncertainty principle for signals):
- **Longer window** → better frequency resolution, worse time resolution (can't track fast changes)
- **Shorter window** → better time resolution, worse frequency resolution

Speech enhancement typically favors a balance around 20-32ms windows since that matches the
timescale over which phonemes are roughly stationary.

---

## 4. Inverse STFT (ISTFT) and Perfect Reconstruction

To go from spectrogram back to waveform, we apply **inverse FFT** to each frame, then
**overlap-add** the frames back together using the hop size. For this reconstruction to be
mathematically lossless, the window/hop combination must satisfy the **Constant Overlap-Add
(COLA)** condition — i.e., the sum of shifted, overlapping windows must equal a constant (or 1)
at every sample. Hann windows with 50-75% overlap naturally satisfy this.

**This is why the sanity check in the README is important**: if your STFT/ISTFT round-trip
doesn't reconstruct the original audio near-perfectly, your window/hop parameters violate COLA
and everything downstream (mask multiplication, model training) will be built on a broken
foundation.

---

## 5. Why We Usually Keep the Noisy Phase

A subtlety in speech enhancement: humans are far less sensitive to phase distortion than to
magnitude distortion. Classical results (and most deep learning literature) show that:
- Estimating **magnitude** accurately gives most of the perceptual improvement.
- Estimating **phase** correctly is much harder (phase is far less structured/predictable than
  magnitude), and using the **noisy phase unchanged** with an enhanced magnitude still sounds
  quite good.

This is why the simplest and most common pipeline is:
```
Enhanced magnitude = Model(noisy magnitude)
Enhanced complex spectrum = Enhanced magnitude * e^(j * noisy phase)
```

More advanced methods (Complex Ratio Masks, Deep Complex U-Net, Conv-TasNet) do try to also
correct the phase, generally giving further improvements — this is a good "stretch goal" to
mention as future work.

---

## 6. Masking Approaches

Instead of predicting the clean magnitude spectrum directly, most modern systems predict a
**mask** — a matrix of values (same shape as the spectrogram) that is multiplied element-wise
with the noisy magnitude to suppress noise-dominated T-F bins and preserve speech-dominated ones.

### Common mask types

- **Ideal Binary Mask (IBM):** 1 where speech energy dominates noise in a T-F bin, 0 otherwise.
  Simple but causes musical noise artifacts (abrupt on/off switching).
- **Ideal Ratio Mask (IRM):** a soft (0–1) value representing the ratio of speech energy to
  total energy in each T-F bin — smoother, better perceptual quality than IBM.
  ```
  IRM(m,k) = sqrt( S(m,k)^2 / (S(m,k)^2 + N(m,k)^2) )
  ```
- **Complex Ratio Mask (CRM):** extends masking to the complex domain, correcting both
  magnitude and phase.

Our project uses a **soft mask (IRM-like)** predicted by a neural network and applied to the
magnitude spectrogram — a good balance of simplicity and quality.

---

## 7. Why a U-Net Architecture?

A spectrogram is structurally similar to an image: two spatial-like axes (frequency, time) with
local correlations (energy in nearby T-F bins tends to be correlated — harmonics, formants,
noise bands, etc.).

**U-Net**, originally designed for image segmentation, is well suited here because:
- The **encoder** (downsampling path) progressively extracts higher-level, larger-receptive-field
  features — capturing broader spectral/temporal patterns (e.g., harmonic structure, noise bands).
- The **decoder** (upsampling path) reconstructs a full-resolution output (the mask) from these
  features.
- **Skip connections** between encoder and decoder let fine-grained, low-level spectral details
  (important for preserving speech clarity) flow directly to the output, rather than being
  lost through the bottleneck.

This mirrors exactly why U-Net works well for image segmentation: masking a spectrogram is
conceptually similar to segmenting an image into "speech" vs "not speech" regions, but with a
continuous (soft) rather than binary output.

### Why predict a mask instead of predicting clean magnitude directly?

- Masking constrains the output to be a *scaled version of the input*, which is an easier,
  more stable learning problem than generating an entirely new spectrum from scratch.
- It also guarantees the output magnitude can never exceed the noisy input magnitude (a
  reasonable physical constraint, since we're removing noise, not adding new energy).

---

## 8. Loss Functions — Why They Matter

### 8.1 MSE on magnitude
Simplest option — directly penalizes squared error between predicted and clean magnitude
spectrograms. Problem: speech magnitude spans a huge dynamic range (loud voiced segments vs
quiet fricatives), so raw MSE is dominated by loud regions and undervalues quiet ones.

### 8.2 Log-magnitude MSE
Taking `log(1 + magnitude)` compresses the dynamic range, similar to how human hearing
perceives loudness roughly logarithmically (related to the psychoacoustic **Weber-Fechner
law**). This makes the loss pay more balanced attention to both loud and quiet speech
components — generally improves perceptual quality over raw MSE.

### 8.3 SI-SDR (Scale-Invariant Signal-to-Distortion Ratio)
Rather than comparing spectrograms, SI-SDR is computed **directly on the reconstructed
waveform**, and measures how much of the output signal explains the target signal versus how
much is left-over "error" energy, while being invariant to the overall output scale
(amplitude). It's defined as:

```
s_target = ( <ŝ, s> / ||s||² ) * s
e_noise  = ŝ - s_target
SI-SDR   = 10 * log10( ||s_target||² / ||e_noise||² )
```

This metric correlates much more closely with human-perceived speech quality than plain MSE,
which is why most modern papers optimize it directly (as a loss, using `-SI-SDR` since we
minimize loss but want to maximize SI-SDR).

**Practical tip:** many strong systems use a **combination**: log-magnitude loss for stable
early training + SI-SDR loss (computed after ISTFT) for perceptually-aligned fine-tuning.

---

## 9. Evaluation Metrics — Why These Specific Ones

Speech enhancement quality can't be judged by loss value alone — we need metrics that
correlate with human perception and are standard in the literature (so your results are
comparable to published work):

- **PESQ (Perceptual Evaluation of Speech Quality):** an ITU-T standard algorithm that models
  human auditory perception to give a score (roughly 1–4.5) estimating perceived speech
  quality. Widely used and is the closest common proxy to "would a human think this sounds
  clean."
- **STOI (Short-Time Objective Intelligibility):** estimates how *intelligible* (understandable)
  speech is, based on correlation of short-time temporal envelopes between clean and enhanced
  signal. Important because a signal can sound "denoised" but lose intelligibility (words
  become harder to make out) — STOI catches this failure mode that PESQ alone might miss.
  Score range 0–1 (higher = more intelligible).
- **SI-SDR:** as above — a numerically simple, scale-invariant measure of how well the target
  signal is reconstructed versus residual error.

Reporting all three (rather than just one) gives a fuller picture: PESQ for perceived quality,
STOI for intelligibility, SI-SDR for signal-level accuracy.

---

## 10. Classical Baseline: Spectral Subtraction

Before comparing your neural model's results, it's important to show it beats a classical DSP
method — this both validates that your AI model is adding real value and shows depth of
understanding of the field's history.

**Spectral subtraction** assumes:
1. Noise is roughly stationary (its spectral shape doesn't change much over time).
2. We can estimate the noise spectrum from a noise-only segment (e.g., the first few frames of
   an utterance, often silence/background before speech starts).
3. We can then simply subtract this estimated noise magnitude from the noisy magnitude at every
   frame:
   ```
   Ŝ(m,k) = X(m,k) - α * N̂(k)
   ```
   where `α` is an over-subtraction factor, and a **spectral floor** `β` prevents magnitudes
   from going negative (which would be non-physical), typically:
   ```
   Ŝ(m,k) = max( X(m,k) - α*N̂(k), β*X(m,k) )
   ```

**Limitations (which is exactly why neural models help):**
- Assumes noise is stationary — fails badly on non-stationary noise (traffic, babble, music).
- Produces "**musical noise**" artifacts — isolated, randomly appearing/disappearing T-F bins
  that sound like short "blips" or "musical" tones, from imperfect noise estimation.
- Cannot adapt to the specific characteristics of speech vs. noise the way a learned model can.

---

## 11. Time-Domain Alternative: Conv-TasNet (for stretch goal)

Instead of STFT → mask → ISTFT, **Conv-TasNet** replaces the fixed STFT/ISTFT transform with a
**learned encoder/decoder**:
- A 1D convolutional "encoder" learns its own basis functions (instead of Fourier basis) to
  transform raw waveform into a latent representation.
- A **Temporal Convolutional Network (TCN)** with dilated convolutions estimates a mask in this
  learned latent space (dilated convolutions give a large receptive field efficiently, capturing
  long-range temporal dependencies in speech).
- A "decoder" (transposed convolution) converts the masked representation back to a waveform.

**Why it can outperform STFT-based methods:** the learned basis can be optimized end-to-end for
the enhancement task specifically, rather than using the generic, task-agnostic Fourier basis —
and it sidesteps the phase estimation problem entirely since it never separates magnitude/phase
in the first place.

This is a good "Option C" to implement after the U-Net baseline works, to directly compare
T-F domain vs time-domain approaches in your write-up.

---

## 12. Practical/Data Considerations

- **Variable-length audio:** real utterances have different durations → different numbers of
  STFT frames. Handle this by either (a) padding/cropping all spectrograms to a fixed number of
  time frames per batch, or (b) using variable-length batching with padding + masking in the
  loss computation.
- **Normalization:** magnitude spectrograms benefit from log-compression (see §8.2) and often
  per-utterance or per-dataset mean/variance normalization to stabilize training.
- **SNR variety in training data:** training on a range of SNR levels (mix of easy and hard
  noisy conditions) helps the model generalize better than training only on one fixed noise
  level.
- **Train/test noise mismatch:** a good experiment is testing on noise types *not seen* during
  training, to check how well the model generalizes rather than just memorizing training noise
  characteristics.

---

## 13. Summary: What This Project Demonstrates

| Skill | Where it shows up |
|---|---|
| Classical DSP | STFT/ISTFT design, windowing trade-offs, spectral subtraction baseline |
| Statistical signal modeling | Noise estimation, masking theory (IBM/IRM) |
| Deep learning architecture design | U-Net for T-F masking, understanding of skip connections/receptive fields |
| Loss function design | Log-magnitude loss, SI-SDR loss, understanding perceptual vs numerical objectives |
| Rigorous evaluation | PESQ, STOI, SI-SDR — standard, comparable metrics |
| Systems thinking | End-to-end pipeline from raw waveform to enhanced waveform, deployable demo |

This combination — classical DSP understanding **plus** deep learning modeling **plus**
proper perceptual evaluation — is exactly what distinguishes a "signal processing + AI"
specialist from someone who just trained a generic model on audio data.