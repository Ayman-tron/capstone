import numpy as np
import pandas as pd
from scipy.fft import fft
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import os

directory = "actual_test"
num = len(os.listdir(os.path.join(directory,os.listdir(directory)[0]))) # returns number of test samples in each sensor folder

def main():
    
    freqCnt = calcDomFreqCnt()
    row = makeDataRow(num-1,freqCnt)
    print(row)
    
    ###### Uncomment this to update the training test csv with more data
    # ML_data = makeTrainingData()
    # pd.DataFrame(ML_data).to_csv('ML_trainingTest2.csv')


def makeDataRow(fileID, FreqCnt):
        row = np.zeros((FreqCnt * 2 * 4) + 1)

        for fnum in range(0,len(os.listdir(directory))): # for each sensor A, C, G, I
            filename = os.listdir(directory)[fnum]
            f = os.path.join(directory, filename)
            f = os.path.join(f,os.listdir(f)[fileID])
            data_raw = read_data(f)
            data_peaks = (getFrequencyPeaks(data_raw[0],data_raw[1],data_raw[2]))
            for n in range(0, len(data_peaks)):
                row[n * 2 + fnum * FreqCnt * 2] = data_peaks[n][0]
                row[n * 2 + fnum * FreqCnt * 2 + 1] = data_peaks[n][1]
            # print(row)
        
        return row

def calcDomFreqCnt():
    j = 0
    DomFreqCnt = 0
    while j < num:
        for filename in os.listdir(directory):
            f = os.path.join(directory, filename)
            f = os.path.join(f,os.listdir(f)[j])
            print(f)
            data_raw = read_data(f)
            data_peaks = (getFrequencyPeaks(data_raw[0],data_raw[1],data_raw[2]))
            # print(len(data_fft))
            DomFreqCnt = max(DomFreqCnt, len(data_peaks))
        j += 1
    return DomFreqCnt

def makeTrainingData():
    # Checks the largest amount of significant frequencies from all data sets
    maxDomFreqCnt = calcDomFreqCnt()

    # Loop through each csv containing test samples and convert them all into ML training data
    ML_data = []
    binaryID = 0
    for sample in range(0, num): # for each sample...
        data_row = makeDataRow(sample, maxDomFreqCnt)        
        if sample % 3 == 0 and not sample == 0: 
            if binaryID == 0:
                binaryID = 1
            else:
                binaryID = binaryID * 10
        data_row[-1] = binaryID
        ML_data.append(data_row)
    return ML_data

# Function for reading the data from csv files
def read_data(path):
    # Read the data from the CSV file
    data = pd.read_csv(path)

# Extract the acceleration data from the CSV file
    accel_x = data.iloc[:, 1].values
    accel_y = data.iloc[:, 2].values
    accel_z = data.iloc[:, 3].values
    return [accel_x, accel_y, accel_z]

# Function for getting the dominant frequencies from frequency domain with windowing
def getFrequencyPeaks(x,y,z):
    # Calculate the magnitude of the acceleration using Euclidean norm
    print(type(x[0]))
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
    
    # Only output values with amplitudes higher than a calculated output
    threshold = 0.1 * sorted_dominant_frequencies_amplitudes[0][1]    
    out = []
    for i in sorted_dominant_frequencies_amplitudes:
        if i[1] >= threshold:
            out.append(i)
    return out



if __name__ == "__main__":
    main()