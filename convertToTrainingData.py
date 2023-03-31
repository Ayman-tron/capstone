import numpy as np
import pandas as pd
from scipy.fft import fft
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import os

# Function for reading the data from csv files
def read_data(path):
    # Read the data from the CSV file
    data = pd.read_csv(path)

# Extract the acceleration data from the CSV file
    accel_x = data.iloc[:, 1].values
    accel_y = data.iloc[:, 2].values
    accel_z = data.iloc[:, 3].values

    return [accel_x, accel_y, accel_z]


# Function for getting the dominant frequencies from frequency domain
# Performs windowing, 
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
    threshold = 0.1 * sorted_dominant_frequencies_amplitudes[0][1]
    # Print the dominant frequencies and corresponding amplitudes above the threshold
    #print("Dominant frequencies and amplitudes:")
    
    out = []
    for i in sorted_dominant_frequencies_amplitudes:
        if i[1] >= threshold:
            out.append(i)

    return out


directory = "actual_test"
num = len(os.listdir(os.path.join(directory,os.listdir(directory)[0]))) # returns number of test samples in each sensor folder

# Checks the largest amount of significant frequencies from all data sets
j = 0
maxDomFreqCnt = 0
while j < num:
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        f = os.path.join(f,os.listdir(f)[j])
        data_raw = read_data(f)
        data_fft = (getFrequencyPeaks(data_raw[0],data_raw[1],data_raw[2]))
        # print(len(data_fft))
        maxDomFreqCnt = max(maxDomFreqCnt, len(data_fft))
    j += 1
print(maxDomFreqCnt)

# Loop through each csv containing test samples and convert them all into ML training data
# i = 0
ML_data = []
binaryID = 0
for i in range(0, num):
    data_row = np.zeros((maxDomFreqCnt * 2 * 4) + 1)
    # print(len(os.listdir(directory)))
    for fnum in range(0,len(os.listdir(directory))):
        filename = os.listdir(directory)[fnum]
        f = os.path.join(directory, filename)
        f = os.path.join(f,os.listdir(f)[i])
        data_raw = read_data(f)
        data_fft = (getFrequencyPeaks(data_raw[0],data_raw[1],data_raw[2]))
        for n in range(0, len(data_fft)):
            data_row[n * 2 + fnum * maxDomFreqCnt * 2] = data_fft[n][0]
            data_row[n * 2 + fnum * maxDomFreqCnt * 2 + 1] = data_fft[n][1]
        # print(data_row)
    if i % 3 == 0 and not i == 0:
        if binaryID == 0:
            binaryID = 1
        else:
            binaryID = binaryID * 10
    data_row[-1] = binaryID
    ML_data.append(data_row)
# ML_data = np.array(ML_data)
# print(ML_data)
# print(len(ML_data))
pd.DataFrame(ML_data).to_csv('ML_trainingTest.csv')
   

# print((sorted_dominant_frequencies_amplitudes))

# Plot the amplitude spectrum

# plt.plot(frequency_bins[2:len(frequency_bins)//2],
#          amplitude_spectrum[2:len(amplitude_spectrum)//2])
# plt.xlabel("Frequency (Hz)")
# plt.ylabel("Amplitude (m/s^2)")
# plt.title("Amplitude Spectrum")
# plt.ioff()
# plt.show(block=True)
