# This program calculates and plots the FFT for each axis (x, y, and z) separately

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read the data from the CSV file
data = pd.read_csv("digital_twin\\2000_pipline_test_other_flange_loose.csv")

# Extract the acceleration data from the CSV file
accel_x = data.iloc[:, 1].values
accel_y = data.iloc[:, 2].values
accel_z = data.iloc[:, 3].values

# Apply Hanning window
window = np.hanning(len(accel_x))
windowed_x = accel_x * window
windowed_y = accel_y * window
windowed_z = accel_z * window

# Compute the Fourier transform of the windowed acceleration data
sampling_rate = 232  # Replace with your actual sampling rate
freq_x = np.fft.fftfreq(len(windowed_x), d=1/sampling_rate)
fft_x = np.fft.fft(windowed_x)
freq_y = np.fft.fftfreq(len(windowed_y), d=1/sampling_rate)
fft_y = np.fft.fft(windowed_y)
freq_z = np.fft.fftfreq(len(windowed_z), d=1/sampling_rate)
fft_z = np.fft.fft(windowed_z)

# Find the index corresponding to half the length of the data array
half_data_length = len(freq_x) // 2

# Plot the frequency spectrum of the acceleration data
fig, ax = plt.subplots(3, 1, figsize=(8, 8))
ax[0].plot(freq_x[:half_data_length], np.abs(fft_x[:half_data_length]))
ax[0].set_xlabel("Frequency (Hz)")
ax[0].set_ylabel("Amplitude")
ax[0].set_title("X-axis acceleration")
ax[1].plot(freq_y[:half_data_length], np.abs(fft_y[:half_data_length]))
ax[1].set_xlabel("Frequency (Hz)")
ax[1].set_ylabel("Amplitude")
ax[1].set_title("Y-axis acceleration")
ax[2].plot(freq_z[:half_data_length], np.abs(fft_z[:half_data_length]))
ax[2].set_xlabel("Frequency (Hz)")
ax[2].set_ylabel("Amplitude")
ax[2].set_title("Z-axis acceleration")
plt.tight_layout()
plt.show()
