"""EEG Signal Analyzer (NumPy).

Demonstrates core NumPy operations on a small, hand-written multi-channel
EEG-like signal: reshaping, broadcasting calibration, stacking, splitting,
and per-channel statistics.

NOTE: the data below is SAMPLE (synthetic) data written by hand for
learning purposes. It is not a real EEG recording.
"""

import numpy as np

SAMPLING_RATE_HZ = 1  # assumption: 1 sample per second, so 8 columns = 8 seconds

# 1. Raw data (synthetic): channels 1 and 2 arrive as one flat array
eeg_flat = np.array([12, 15, 14, 18, 20, 22, 19, 17, 85, 88, 90, 86, 79, 81, 83, 80])
channel_3 = np.array([45, 47, 46, 48, 50, 49, 51, 52])

# 2. Reshape the flat array into Channel 1 and Channel 2 (2 rows, 8 columns)
eeg_grid = eeg_flat.reshape(2, 8)

# 3. Calibration: remove the measured baseline offset of each sensor.
#    Channel 1 reads 2 units too low, channel 2 reads 3 units too high,
#    so we subtract the offsets (broadcast: shape (2, 1) over shape (2, 8)).
sensor_offsets = np.array([[-2], [3]])
eeg_clean = eeg_grid - sensor_offsets

# 4. Add Channel 3 as a third row (it is assumed to be already calibrated)
eeg_clean_3ch = np.vstack([eeg_clean, channel_3])

# 5. Split all channels into the first half and the last half of the recording
part_one_eeg, part_two_eeg = np.split(eeg_clean_3ch, 2, axis=1)
half_seconds = part_one_eeg.shape[1] // SAMPLING_RATE_HZ

# 6. Per-channel statistics (axis=1 -> one value per channel)
means = eeg_clean_3ch.mean(axis=1)
stds = eeg_clean_3ch.std(axis=1)
peaks = eeg_clean_3ch.max(axis=1)
peak_times = eeg_clean_3ch.argmax(axis=1) / SAMPLING_RATE_HZ

print("Per-channel statistics (calibrated data)")
for i in range(eeg_clean_3ch.shape[0]):
    print(
        f"Channel {i + 1:02d} | mean = {means[i]:6.2f} | std = {stds[i]:5.2f} | "
        f"peak = {peaks[i]} at t = {peak_times[i]:.0f} s"
    )

print(f"\nFirst {half_seconds} seconds")
for i in range(part_one_eeg.shape[0]):
    print(f"Channel {i + 1:02d} : {part_one_eeg[i]}")

print(f"\nLast {half_seconds} seconds")
for i in range(part_two_eeg.shape[0]):
    print(f"Channel {i + 1:02d} : {part_two_eeg[i]}")
