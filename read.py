import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read the data from the CSV file
data = pd.read_csv("digital_twin\\acceleration_data.csv")

# Extract the acceleration data from the CSV file
accel_x = data.iloc[:, 1].values
accel_y = data.iloc[:, 2].values
accel_z = data.iloc[:, 3].values

# Compute the Fourier transform of the acceleration data
freq_x = np.fft.fftfreq(len(accel_x), 1.0/len(accel_x))
fft_x = np.fft.fft(accel_x)
freq_y = np.fft.fftfreq(len(accel_y), 1.0/len(accel_y))
fft_y = np.fft.fft(accel_y)
freq_z = np.fft.fftfreq(len(accel_z), 1.0/len(accel_z))
fft_z = np.fft.fft(accel_z)

# Plot the frequency spectrum of the acceleration data
fig, ax = plt.subplots(3, 1, figsize=(8, 8))
ax[0].plot(freq_x, np.abs(fft_x))
ax[0].set_xlabel("Frequency (Hz)")
ax[0].set_ylabel("Amplitude")
ax[0].set_title("X-axis acceleration")
ax[1].plot(freq_y, np.abs(fft_y))
ax[1].set_xlabel("Frequency (Hz)")
ax[1].set_ylabel("Amplitude")
ax[1].set_title("Y-axis acceleration")
ax[2].plot(freq_z, np.abs(fft_z))
ax[2].set_xlabel("Frequency (Hz)")
ax[2].set_ylabel("Amplitude")
ax[2].set_title("Z-axis acceleration")
plt.tight_layout()
plt.show()
