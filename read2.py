import numpy as np
import pandas as pd
from scipy.fft import fft
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import os





def read_data(path):
    # Read the data from the CSV file
    data = pd.read_csv(path)

# Extract the acceleration data from the CSV file
    accel_x = data.iloc[:, 1].values
    accel_y = data.iloc[:, 2].values
    accel_z = data.iloc[:, 3].values

    return [accel_x, accel_y, accel_z]



def getFrequencyPeaks(x,y,z):
    # Calculate the magnitude of the acceleration using Euclidean norm
    magnitude = np.sqrt(x**2 + y**2 + z**2)

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
    threshold = 0.15 * sorted_dominant_frequencies_amplitudes[0][1]
    # Print the dominant frequencies and corresponding amplitudes above the threshold
    #print("Dominant frequencies and amplitudes:")
    
    out = []
    for i in range(0, len(sorted_dominant_frequencies_amplitudes)):
        if sorted_dominant_frequencies_amplitudes[i][1] >= threshold:
            out.append(sorted_dominant_frequencies_amplitudes[i])

    return out


directory = "actual_test"
sensor_dirs = os.listdir(directory)
num = len(os.listdir(os.path.join(directory,sensor_dirs[0]))) # returns number of samples in each sensor folder
i = 0
ML_data = []
while i < num:
    data_row = []
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        f = os.path.join(f,os.listdir(f)[i])
        data_raw = read_data(f)
        data_fft = getFrequencyPeaks(data_raw[0],data_raw[1],data_raw[2])
        data_row.append(data_fft)
    i += 1
    ML_data.append(data_row)

print(ML_data[0])
pd.DataFrame(ML_data).to_csv(directory+'test.csv')
   

# print((sorted_dominant_frequencies_amplitudes))

# Plot the amplitude spectrum

# plt.plot(frequency_bins[2:len(frequency_bins)//2],
#          amplitude_spectrum[2:len(amplitude_spectrum)//2])
# plt.xlabel("Frequency (Hz)")
# plt.ylabel("Amplitude (m/s^2)")
# plt.title("Amplitude Spectrum")
# plt.ioff()
# plt.show(block=True)
