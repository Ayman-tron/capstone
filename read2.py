import numpy as np
import pandas as pd
from scipy.fft import fft
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# Read the data from the CSV file
data = pd.read_csv("digital_twin\\2000_pipeline_test.csv")

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

# Get the dominant frequencies and corresponding amplitudes
dominant_frequencies_amplitudes = [
    (frequency_bins[i], amplitude_spectrum[i]) for i in peak_indices]

# Sort by amplitude in descending order
sorted_dominant_frequencies_amplitudes = sorted(
    dominant_frequencies_amplitudes, key=lambda x: x[1], reverse=True)

# Print the 10% highest dominant frequencies and corresponding amplitudes
num_top_frequencies = int(len(sorted_dominant_frequencies_amplitudes) * 0.1)
print("Top 10% dominant frequencies and amplitudes:")
for i in range(num_top_frequencies):
    print(
        f"Frequency: {sorted_dominant_frequencies_amplitudes[i][0]} Hz, Amplitude: {sorted_dominant_frequencies_amplitudes[i][1]}")

# Plot the amplitude spectrum
plt.plot(frequency_bins[2:len(frequency_bins)//2],
         amplitude_spectrum[2:len(amplitude_spectrum)//2])
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude (db)")
plt.title("Amplitude Spectrum")
plt.show()
