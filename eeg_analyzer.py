import numpy as np

# Receive raw data from the EEG headset
eeg_flat = np.array([12, 15, 14, 18, 20, 22, 19, 17, 85, 88, 90, 86, 79, 81, 83, 80])
channel_3 = np.array([45, 47, 46, 48, 50, 49, 51, 52])

# Reshape the flat array into Channel 1 and Channel 2 (2 rows, 8 columns)
eeg_grid = eeg_flat.reshape(2, 8)

# Calibrate EEG sensors by removing the voltage offsets
offsets = np.array([[-2], [3]])
eeg_clean = offsets + eeg_grid

# Add Channel 3 vertically to the cleaned data
eeg_clean_3ch = np.vstack([eeg_clean, channel_3])

# Split all channels into the first 4 seconds and the last 4 seconds
part_one_eeg = np.split(eeg_clean_3ch, 2, axis=1)[0]
part_two_eeg = np.split(eeg_clean_3ch, 2, axis=1)[1]

# Display the processed statistical results and signals
print(f'Channel 01 Mean : {np.mean(eeg_clean[0,:])}\n\
Channel 02 Max Reading : {np.max(eeg_clean[1,:])}\n\
EEG With Three Channels First 4 Seconds\n\
Channel 01 : {part_one_eeg[0,:]}\n\
Channel 02 : {part_one_eeg[1,:]}\n\
Channel 03 : {part_one_eeg[2,:]}\n\
EEG With Three Channels Last 4 Seconds\n\
Channel 01 : {part_two_eeg[0,:]}\n\
Channel 02 : {part_two_eeg[1,:]}\n\
Channel 03 : {part_two_eeg[2,:]}')