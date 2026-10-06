# EEG Signal Analyzer (NumPy)

A small learning project (**Part 1** of my biomedical data science roadmap) that practises core **NumPy** operations on a multi-channel EEG-like signal.

> **Note:** the signal in `eeg_analyzer.py` is a short, hand-written **synthetic sample** (3 channels x 8 samples). It is not a real EEG recording, and no clinical conclusions can be drawn from it.

## What the script does
- **Reshaping:** turns a flat array of values into a `(channels, samples)` grid (`reshape`).
- **Broadcasting calibration:** subtracts a per-channel baseline offset from all samples at once (shape `(2, 1)` against `(2, 8)`).
- **Stacking:** adds a third channel to the calibrated data (`vstack`).
- **Time segmentation:** splits the recording into a first half and a last half (`split`). The sampling rate is assumed to be 1 Hz.
- **Per-channel statistics:** mean, standard deviation, peak value and the time of the peak (`mean`, `std`, `max`, `argmax` with `axis=1`).

## Not included (yet)
Filtering, frequency analysis (FFT, alpha/beta bands), real EEG data and plots. These are the next planned steps (see below).

## Run
```bash
pip install -r requirements.txt
python eeg_analyzer.py
```

## Roadmap
1. Load a real open-source EEG recording (for example from PhysioNet).
2. Apply a band-pass filter and plot the raw vs. filtered signal.
3. Estimate band power (alpha, beta) with an FFT.

## Tech stack
Python 3, NumPy
