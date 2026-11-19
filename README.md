# Acoustic Signal Processing for Parkinson's Screening

A low-cost, non-invasive screening prototype that records a sustained "Ahh"
vowel with an INMP441 I2S MEMS microphone and an ESP32, extracts acoustic
features, and classifies the recording with a Random Forest model.

> **Disclaimer:** This project is a screening aid, not a diagnostic device.
> It is not a substitute for clinical assessment by a qualified professional.

## Overview

The project has two parts:

1. The ESP32 firmware reads 16 kHz, 16-bit audio from the INMP441 microphone
   and streams it to a host computer over USB serial.
2. The host application normalizes the signal, retains voiced frames, extracts
   14 spectral features, and uses a trained Random Forest classifier.

The decision threshold is 56%. A probability at or above that value is reported
as **HIGH RISK (PD)**; lower probabilities are reported as **LOW RISK**.

## Hardware

- ESP32 development board
- INMP441 I2S MEMS microphone
- USB cable

| INMP441 pin | ESP32 GPIO |
|---|---|
| SCK | GPIO 26 |
| WS | GPIO 22 |
| SD | GPIO 21 |

Set the breakout board to left-channel mode (the L/R pin must match
`I2S_CHANNEL_FMT_ONLY_LEFT` in the firmware).

## Software Setup

Python 3.8 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows:

```powershell
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Flash `esp32_inmp441_capture.ino` with the Arduino IDE or PlatformIO. Set the
board's serial port in `par.py`:

```python
SERIAL_PORT = "COM3"  # Use /dev/ttyUSB0 or /dev/ttyACM0 on Linux
```

## Usage

Run the host application from the repository root:

```bash
python par.py
```

The menu provides three options:

1. Score one WAV file.
2. Record eight seconds of sustained "Ahh" audio through the ESP32 and score it.
3. Exit.

Live recordings are saved as `live_recorded_voice.wav`.

## Signal Processing

| Stage | Parameters |
|---|---|
| Acquisition | 16 kHz, 16-bit PCM, 8 seconds |
| Framing | 40 ms frames, 20 ms hop |
| Window | Periodic Hamming window |
| Voice activity | RMS above `max(percentile(rms, 40), 0.015)` |
| Spectrum | Real FFT |
| Feature vector | 14 dimensions |
| Classification | 150-tree Random Forest, balanced classes |

The feature vector contains:

- Mean spectral centroid
- Spectral-centroid standard deviation
- Mean spectral spread
- RMS-volume instability
- Median relative energy for ten spectral sub-bands

## Training

The included `rf_pd_model.pkl` is ready for inference. To retrain it:

```bash
python train.py
```

Training data must remain in these directories:

```text
PD/       Parkinson's recordings
HC/       Healthy-control recordings
PD_TST/   Held-out Parkinson's recordings
HC_TST/   Held-out healthy-control recordings
```

The training script mixes stereo files to mono, peak-normalizes them, extracts
features, and saves a balanced Random Forest classifier.

## Project Structure

```text
.
├── esp32_inmp441_capture.ino
├── par.py
├── train.py
├── rf_pd_model.pkl
├── requirements.txt
├── HC/
├── PD/
├── HC_TST/
├── PD_TST/
└── sample.wav
```

## Evaluation

The original evaluation used five-fold cross-validation on 1,142 sustained-vowel
recordings (560 PD and 582 healthy controls):

| Metric | Value |
|---|---|
| Accuracy | 91.42% +/- 0.81% |
| Sensitivity | 83.57% |
| Specificity | 98.97% |
| Precision | 98.73% |
| F1-score | 90.52% |

These results do not establish clinical validity and do not account for
subject-independent or cross-device effects.

## Limitations

- Screening only; not a diagnosis
- Sustained vowels only
- A recording from another microphone may not match live INMP441 capture
- Cross-validation should be subject-independent before clinical use
- Host-based inference is not yet available directly on the ESP32
- Serial configuration can require adjustment on macOS, Linux, and Windows

## Future Work

- Collect and validate INMP441 recordings
- Add subject-independent cross-validation
- Evaluate pitch, jitter, shimmer, and voice-break features
- Improve noise reduction and normalization
- Tune the decision threshold with ROC analysis
- Add a clinical validation protocol