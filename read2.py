import numpy as np
import pandas as pd
from scipy.fft import fft
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# Read the data from the CSV file
data = pd.read_csv("digital_twin\\data\\2000_pipeline_test.csv")

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

# Calculate the threshold for 10% of the largest peak's magnitude
threshold = 0.1 * sorted_dominant_frequencies_amplitudes[0][1]

# Print the dominant frequencies and corresponding amplitudes above the threshold
print("Dominant frequencies and amplitudes above 10% of the largest peak:")
for freq, amp in sorted_dominant_frequencies_amplitudes:
    if amp >= threshold:
        print(f"Frequency: {freq} Hz, Amplitude: {amp}")
    else:
        break

# Plot the amplitude spectrum
plt.plot(frequency_bins[2:len(frequency_bins)//2],
         amplitude_spectrum[2:len(amplitude_spectrum)//2])
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude (m/s^2)")
plt.title("Amplitude Spectrum")
plt.show()
