import numpy as np
import pandas as pd
from scipy.fft import fft
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# Read the data from the CSV file
data = pd.read_csv("digital_twin\\acceleration_data.csv")

# Extract the acceleration data from the CSV file
accel_x = data.iloc[:, 1].values
accel_y = data.iloc[:, 2].values
accel_z = data.iloc[:, 3].values

# Calculate the magnitude of the acceleration using Euclidean norm
magnitude = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)

# Apply a window function (Hanning window)
window = np.hanning(len(magnitude))
windowed_data = magnitude * window

# Perform the Fast Fourier Transform (FFT)
fft_data = fft(windowed_data)

# Calculate the amplitude spectrum
amplitude_spectrum = 2 * np.abs(fft_data) / len(windowed_data)

# Calculate the frequency bins
sampling_rate = 232  # Replace with your actual sampling rate
frequency_bins = np.fft.fftfreq(len(windowed_data), d=1/sampling_rate)

# Find peaks in the amplitude spectrum
peak_indices, _ = find_peaks(amplitude_spectrum[:len(amplitude_spectrum)//2])

# Print dominant frequencies and corresponding amplitudes
print("Dominant frequencies and amplitudes:")
for i in peak_indices:
    print(
        f"Frequency: {frequency_bins[i]} Hz, Amplitude: {amplitude_spectrum[i]}")

# Plot the amplitude spectrum
plt.plot(frequency_bins[:len(frequency_bins)//2],
         amplitude_spectrum[:len(amplitude_spectrum)//2])
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.title("Amplitude Spectrum")
plt.show()
