# IoT-Based Predictive Maintenance Capstone Project

This repository contains the code for an IoT-based predictive maintenance application using vibration data analysis. The project aims to develop a system that analyzes vibration data from industrial machinery to predict and prevent failures, reducing downtime and maintenance costs.

## Overview

The project consists of two Python scripts that process vibration data collected from IoT devices attached to industrial machinery. The scripts read the data from CSV files, perform Fast Fourier Transform (FFT) to extract frequency information, and analyze the results to identify dominant frequencies that may indicate potential issues.

### Scripts

1. **plotEuclidean.py** - This script reads the vibration data in the x, y, and z directions, calculates the magnitude of the acceleration using the Euclidean norm, applies a Hanning window, and performs FFT to calculate the amplitude spectrum. It then finds and prints the dominant frequencies and their corresponding amplitudes and plots the amplitude spectrum.

2. **read.py** - This script reads the vibration data in the x, y, and z directions, applies a Hanning window, and performs FFT for each axis separately. It then plots the frequency spectrum of the acceleration data for each axis.

## Installation

1. Clone the repository:
`https://github.com/Ayman-tron/capstone.git`

2. Install the required packages:

`pip install numpy pandas scipy matplotlib`


