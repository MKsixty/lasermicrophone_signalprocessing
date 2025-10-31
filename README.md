# Laser Microphone Signal Processing Repository

## Overview

This repository contains the complete implementation of a low-cost laser microphone system capable of remotely capturing and reconstructing intelligible speech from vibrating surfaces through optical sensing. The project demonstrates end-to-end acoustic signal recovery using a 650 nm laser transmitter, photodiode-based receiver with dual transimpedance amplifier (TIA) architecture, and multi-stage digital signal processing pipeline.

**Key Achievement**: 15.7-fold intelligibility improvement (STOI = 0.157) with 15.0% Word Error Rate using Whisper Medium transcription on processed audio captured at 1-3 meter distances.

---

## Repository Structure

```
lasermicrophone_signalprocessing/
├── Analysis.m                    # MATLAB signal processing pipeline
├── Laser_MicrophoneV1.py        # Python GUI application with real-time transcription
├── WERcalc.py                   # Word Error Rate calculation utility
├── recordings/                  # Raw and processed audio samples
├── hardware/                    # PCB schematics, 3D models, BOMs
└── docs/                        # Technical documentation and references
```

---

## Hardware Components

### Optical System
- **Transmitter**: BezosMax 650 nm red laser pointer (>5 mW, 2500 m range)
- **Receiver**: BPW34 silicon PIN photodiode (400-1100 nm, 0.6 A/W responsivity @ 650 nm)
- **Reflective Surface**: 3 mm glass window (8.0% reflectivity, superior to Perspex by 0.53 dB SNR)

### Electronics
- **Dual-TIA Architecture**: MCP6292 rail-to-rail op-amps with 10 kΩ feedback resistors
- **Common-Mode Rejection**: Differential amplifier stage (CMRR enhancement for ambient light suppression)
- **ADC Interface**: HiFi USB sound card (48 kHz, 16-bit resolution)
- **Power**: Single 5V USB supply (<500 mA)

### Mechanical
- Custom 3D-printed laser holder with 360° yaw and 90° roll adjustment
- Black PLA receiver enclosure (12 cm height, minimizes optical reflections)

**Total Hardware Cost**: R902 (under R2000 budget constraint)

---

## Software Implementation

### 1. MATLAB Signal Processing (`Analysis.m`)

Multi-stage enhancement pipeline implementing classical DSP techniques:

#### Stage 1: FIR Band-Pass Filtering
```matlab
filter_order = 300;
f_norm = [100 3400] / (fs/2);
b_bp = fir1(filter_order, f_norm, 'bandpass', blackman(filter_order+1));
x_bp = filtfilt(b_bp, 1, x);  % Zero-phase filtering
```
- **Purpose**: Isolate speech band (100-3400 Hz), suppress low-frequency hum and high-frequency sensor noise
- **Window**: Blackman (superior stopband attenuation vs. Hamming/Hann)
- **Implementation**: Zero-phase forward-backward filtering to preserve temporal alignment

#### Stage 2a: Wiener Filter
```matlab
wiener_gain = max(0, 1 - noise_power ./ (frame_power + eps));
wiener_gain = smoothdata(wiener_gain, 'gaussian', 5);  % Musical noise reduction
```
- **Method**: MMSE estimation with noise profile learned from initial 0.5s silence
- **Frame Size**: 20 ms with 50% overlap (Hann window)
- **Smoothing**: Gaussian kernel (5-bin) to suppress musical noise artifacts

#### Stage 2b: Spectral Subtraction
```matlab
cleaned_mag = frame_mag - alpha * noise_spectrum;
cleaned_mag = max(cleaned_mag, beta * noise_spectrum);  % Spectral floor
```
- **Over-subtraction Factor (α)**: 2.0 (aggressive noise removal)
- **Spectral Floor (β)**: 0.001 (prevents excessive attenuation)
- **Window**: Hamming (20 ms frames, 10 ms hop)

#### Stage 2c: Adaptive Spectral Gating
```matlab
snr_estimate = (frame_mag - noise_mean) ./ (noise_std + eps);
gate = 1 ./ (1 + exp(-sensitivity * (snr_estimate - 1)));  % Sigmoid gating
gate = mean(gate_buffer, 2);  % 3-frame temporal smoothing
```
- **Noise Learning**: First 5 seconds (skips initial 30 ms to avoid startup artifacts)
- **Sensitivity**: 2.5 (sigmoid steepness)
- **Soft Gating**: Logistic function for smooth attenuation (vs. hard thresholding)

#### Stage 3: Adaptive Amplitude Restoration
```matlab
% Voice Activity Detection
is_voice = (zcr > 0.03 && zcr < 0.25 && energy > 0.001);

% Gain calculation from original audio
restoration_gain = original_voice_rms / denoised_voice_rms;
restoration_gain = min(restoration_gain, 8.0);  % Limit over-amplification

% Selective amplification
amplified = signal .* (1 + voice_mask * (restoration_gain - 1));
```
- **VAD Criteria**: ZCR ∈ [0.03, 0.25] and energy > 0.001 (25 ms frames)
- **Gain Limiting**: Max 8× to prevent noise re-amplification
- **Normalization**: 99.5th percentile scaling (robust to outlier spikes)

#### Stage 4: Combined Best Approach
Sequential application: Wiener → Spectral Subtraction → Adaptive Restoration
- **Rationale**: Balances smoothness (Wiener), aggressiveness (SpectralSub), and clarity (VAD-based gain)
- **Result**: STOI = 0.157, SNR(VAD) = 3.384 dB, optimal balance across metrics

**Unique Libraries**:
- `signal` (scipy): Butterworth filter design, sosfilt for stable IIR implementations
- `audiowrite`: 48 kHz WAV export for downstream transcription
- `smoothdata`: Gaussian smoothing for musical noise suppression

---

### 2. Python GUI Application (`Laser_MicrophoneV1.py`)

Real-time interface for audio capture, visualization, and transcription.

#### Core Features

**Audio Acquisition**:
```python
stream = p.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=48000,
    input=True,
    input_device_index=device_index,
    frames_per_buffer=1024
)
```
- **PyAudio Integration**: Direct USB sound card interfacing (48 kHz, 16-bit mono)
- **Buffer Management**: 1024-sample chunks for low-latency streaming

**Real-Time Visualization**:
```python
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
self.spectrogram_canvas = MplCanvas(width=8, height=3, dpi=100)
```
- **Waveform Display**: Time-domain amplitude plot with RMS overlay
- **Spectrogram**: STFT-based frequency-time analysis (updated per frame)
- **Dynamic Scaling**: Auto-adjusting axes for signal monitoring

**Speech-to-Text Integration**:
```python
def convert_audio_to_16khz(self, input_file):
    command = ["ffmpeg", "-i", input_file, "-ar", "16000", "-ac", "1", output_file]
    subprocess.run(command, check=True)
```
- **FFmpeg Preprocessing**: Automatic 16 kHz mono conversion for ASR compatibility
- **Multi-Model Support**: 
  - **Vosk** (offline): Small (40 MB), Medium (128 MB), Large (1.5 GB) English models
  - **Google Speech Recognition** (online): Cloud-based via `speech_recognition` library
  - **Whisper** (offline): Small (244 MB), Medium (769 MB) models via `openai-whisper`

**Transcription Workflow**:
```python
model = whisper.load_model('medium')
result = model.transcribe(audio_file, fp16=False)
transcription = result['text'].strip()
```

**GUI Framework**:
- **PyQt6**: Native cross-platform interface with QMediaPlayer for audio playback
- **Tabs**: Recordings management, visualization, transcription settings
- **Configuration**: Device selection, filter parameters, model switching

**Unique Libraries**:
- `pyaudio`: Cross-platform audio I/O (PortAudio wrapper)
- `vosk`: Offline Kaldi-based speech recognition (no internet required)
- `openai-whisper`: Transformer-based transcription (multilingual, robust to noise)
- `speech_recognition`: Google Speech API interfacing
- `matplotlib` + `Qt5Agg`: Embedded plotting in PyQt applications
- `PyQt6.QtMultimedia`: QMediaPlayer/QAudioOutput for WAV playback

---

### 3. WER Calculation (`WERcalc.py`)

Evaluates transcription accuracy using standard Word Error Rate metric:

```python
def calculate_wer(reference, hypothesis):
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()
    
    # Levenshtein distance via dynamic programming
    d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1))
    # ... matrix computation ...
    
    wer = (substitutions + deletions + insertions) / len(ref_words) * 100
    return wer
```

**Formula**:
$$
\text{WER} = \frac{S + D + I}{N} \times 100\%
$$
where S = substitutions, D = deletions, I = insertions, N = reference word count.

**Usage**:
```python
reference = "the quick brown fox jumps"
hypothesis = "the quack brown fox jumped"
wer = calculate_wer(reference, hypothesis)  # Returns WER percentage
```

---

## Performance Metrics

### Signal Quality (1m Distance, Glass Window)

| Metric | Raw Recording | Combined Processing | Improvement |
|--------|---------------|---------------------|-------------|
| **STOI** | 0.010 | 0.157 | **15.7× increase** |
| **SNR (VAD)** | 5.305 dB | 3.384 dB | Balanced (post-filtering) |
| **SNR (Noise Floor Est.)** | 14.341 dB | 15.826 dB | +1.485 dB |

### Transcription Accuracy

| ASR Model | Clean Audio WER | Raw Audio WER | Processed WER | Accuracy Gain |
|-----------|-----------------|---------------|---------------|---------------|
| **Whisper Medium** | 9.09% | 16.46% | **15.0%** | +8.9% absolute |
| **Vosk Large** | 31.2% | 54.79% | 19.16% | +64.98% relative |
| **Google Speech API** | 51.35% | Failed | Failed | Incompatible |

**Key Finding**: Additional Audacity noise reduction (15-17 dB) degraded Whisper Medium performance (21.13% WER), confirming optimal balance achieved by multi-stage pipeline.

### Range Sensitivity

| Distance | SNR (VAD) | STOI | WER (Combined) | Interpretation |
|----------|-----------|------|----------------|----------------|
| 1m | 3.384 dB | 0.157 | 15.0% | **Optimal perceptual quality** |
| 2m | 7.531 dB | 0.153 | 15.18% | Balanced |
| 3m | 8.301 dB | 0.085 | 8.11% | **Best ASR (context-driven)** |

**Non-linear Observation**: Higher SNR at 3m due to mean optical power preservation, but STOI degrades from optical diffraction and beam coherence loss. Whisper's language model compensates via contextual inference (lowest WER despite poorest intelligibility).

---

## Installation & Usage

### MATLAB Environment

**Requirements**:
```matlab
% Required Toolboxes:
% - Signal Processing Toolbox
% - Audio Toolbox (for STOI calculation)
```

**Run Processing Pipeline**:
```matlab
% Load recorded audio
[x, fs] = audioread('recordings/1m_glass_raw.wav');

% Load clean reference for SNR/STOI calculation
[clean_ref, ~] = audioread('recordings/clean_reference.wav');

% Execute full pipeline
Analysis;  % Generates stage_1_bandpass.wav through stage_4_combined.wav
```

**Output Files**:
- `stage_1_bandpass.wav`: After FIR filtering (100-3400 Hz)
- `stage_2a_wiener_filtered.wav`: Wiener denoising
- `stage_2b_spectral_subtraction.wav`: Spectral subtraction
- `stage_2c_adaptive_gating.wav`: Adaptive gating
- `stage_3_*_amplified.wav`: Voice-amplified versions
- `stage_4_combined.wav`: **Final optimized output**

### Python Application

**Dependencies**:
```bash
pip install PyQt6 pyaudio numpy matplotlib scipy vosk openai-whisper speech_recognition
```

**FFmpeg Requirement** (for Whisper compatibility):
- **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html), add to PATH
- **Linux**: `sudo apt install ffmpeg`
- **macOS**: `brew install ffmpeg`

**Launch GUI**:
```bash
python Laser_MicrophoneV1.py
```

**Workflow**:
1. **Device Selection**: Choose USB sound card input/output from dropdown
2. **Recording**: Click "Start Recording" → Laser microphone captures audio
3. **Visualization**: Monitor waveform/spectrogram in real-time
4. **Upload Processed Audio**: Load `stage_4_combined.wav` from MATLAB output
5. **Transcription**: Select ASR model (Vosk Large/Whisper Medium recommended) → "Start Transcription"
6. **Export**: Save transcription as `.txt` file

---

## Key Algorithmic Innovations

### 1. Dual-TIA Common-Mode Rejection

**Problem**: Ambient light introduces DC photocurrent offsetting amplifier headroom.

**Solution**: Two TIAs process same photodiode output in opposite polarities. Differential amplifier subtracts common-mode (ambient) while doubling modulated signal:

$$
V_{\text{diff}} = G_d \cdot (V_{\text{TIA1}} - V_{\text{TIA2}}) = G_d \cdot 2 \cdot I_{\text{mod}} \cdot R_f
$$

**Result**: >10 dB ambient noise suppression (CMRR-dependent).

### 2. Hybrid SNR Estimation

**Three Complementary Methods**:

- **Method 1 (Reference-Based)**: Segmented VAD on clean reference, measures residual noise in silence regions after gain compensation
  ```matlab
  snr_ref = 10*log10(signal_pwr_speech / noise_pwr_silence)
  ```

- **Method 2 (Noise Floor)**: First 0.5s assumed noise-only
  ```matlab
  snr_est = 10*log10(mean(signal.^2) / mean(noise.^2))
  ```

- **Method 3 (VAD-Based)**: Energy-based speech/noise classification per frame
  ```matlab
  snr_vad = 10*log10(mean_energy_voice / mean_energy_noise)
  ```

**Insight**: VAD-based SNR most diagnostically valuable—revealed individual filtering stages reduced SNR (1.9-2.6 dB) before adaptive amplification recovered to 3.2-3.9 dB.

### 3. Percentile-Based Normalization

**Problem**: Absolute peak normalization penalizes entire signal for transient spikes.

**Solution**: Use 99.5th percentile of voice regions (identified via VAD) as normalization reference:

```matlab
sorted_voice_abs = sort(abs(voice_signal));
peak_voice_99 = sorted_voice_abs(round(0.995 * length(sorted_voice_abs)));
normalized = signal / peak_voice_99 * 0.98;
```

**Advantage**: Robust to outlier spikes, preserves speech dynamics while preventing clipping.

---

## Hardware Design Files

### PCB Schematics (`hardware/`)
- **KiCad Project**: Dual-TIA receiver with configurable jumpers (JP1-JP4, JPFilter1/2)
- **Gerber Files**: Ready for fabrication (55 mm × 42 mm, 2-layer)
- **Test Points**: TP1-TP6 for oscilloscope debugging

### 3D Models (`hardware/STL/`)
- **Laser Holder**: Base stand, 360° rotor, 90° pitch adjustment
- **Receiver Enclosure**: Black PLA (12 cm height, 7.5 mm photodiode aperture)

### Bill of Materials
- **Electronics BOM**: R902 total (photodiode, op-amps, PCB, USB components)
- **Test Box BOM**: R215 (plywood, glass pane, contact adhesive)

---

## Validation & Reproducibility

### Controlled Test Setup
- **Enclosure**: 20×17.8×25 cm chipboard sound box (sealed with Genkem adhesive)
- **Speaker**: JBL Go 3 (110 Hz - 20 kHz, >85 dB SNR, max volume)
- **Test Audio**: 2-minute monologue from film "Fences" (included in `recordings/clean_reference.wav`)
- **Distances**: 1m, 2m, 3m with fixed alignment

### Objective Metrics
- **STOI**: MATLAB `stoi()` function (0-1 scale, >0.75 = good intelligibility)
- **WER**: Levenshtein distance-based calculation (`WERcalc.py`)
- **SNR**: Three-method validation (reference, noise floor, VAD-based)

---

## Limitations & Future Work

### Current Constraints
- **Range**: 1-3 m operational (optimal 1m for human listening, 2-3m for transcription)
- **Alignment**: Manual beam steering required (no auto-tracking)
- **Processing**: Offline only (MATLAB-based, no real-time embedded DSP)
- **Bandwidth**: 3.4 kHz cutoff excludes sibilants and high-frequency nuances

### Recommended Enhancements
1. **Optical System**:
   - MEMS mirrors or galvanometer steering for adaptive alignment
   - 1550 nm eye-safe laser with silicon APD detection (higher permissible power)

2. **Real-Time Processing**:
   - STM32H7 microcontroller with ARM CMSIS-DSP library
   - Lightweight RNNoise integration for embedded denoising

3. **Advanced Algorithms**:
   - SEGAN (Speech Enhancement GAN) for non-stationary noise
   - Whisper fine-tuning on laser microphone recordings

4. **Environmental Robustness**:
   - Automated calibration via internal reference tones
   - Humidity/temperature compensation for long-term deployment

---

## Citation & Licensing

**Academic Use**:
```bibtex
@misc{mokhakala2025lasermicrophone,
  author = {Mokhakala, Molise},
  title = {Laser-Based Listening Device: Low-Cost Optical Acoustic Sensing with Multi-Stage DSP},
  year = {2025},
  institution = {University of Cape Town, Department of Electrical Engineering},
  url = {https://github.com/MKsixty/lasermicrophone_signalprocessing}
}
```

**License**: MIT (hardware schematics, software code)  
**Hardware Designs**: Open-source (KiCad, STL files freely distributable)

---

## Contact & Support

**Author**: Molise Mokhakala  
**Institution**: University of Cape Town, BSc (Eng) Mechatronics Engineering  
**Supervisor**: Dr. Stephen Paine  

**Issues**: Report bugs or feature requests via GitHub Issues  
**Contributions**: Pull requests welcome for algorithm improvements, ASR model integrations, or hardware optimizations

---

**Acknowledgments**:  
Special thanks to Tsepang (Power Lab technician) for test box fabrication, and Dr. Paine for continuous technical guidance throughout the project.
