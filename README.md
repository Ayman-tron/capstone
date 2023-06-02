# IoT-Based Predictive Maintenance Capstone Project

This repository contains the code for an IoT-based predictive maintenance application using vibration data analysis. The project aims to develop a system that identifies faults in pipe systems by analyzing acceleration data from IoT devices using a k-nearest neighbor classification machine learning algorithm.

## Overview

This project uses a series of Python scripts to form a pipeline for data acquisition, preprocessing, feature extraction, and fault detection. The scripts read the data from CSV files, perform Fast Fourier Transform (FFT) to extract frequency information, apply k-nearest neighbor (k-NN) classification, and communicate the results for real-time monitoring and analysis of the system's health status.

## Scripts

1. `fault_detect.py` - This is the main script that handles the machine learning aspect of the project. It employs the k-NN algorithm to predict whether a flange is healthy or not based on the processed sensor data. 

2. `convertToTrainingData.py` - This script is responsible for processing raw sensor data and converting it into a suitable format for machine learning.

3. `plotEuclidean.py` - This script generates a visualization of the Euclidean distance between the sensor data points.

4. `plotFFT.py` - This script performs Fast Fourier Transform (FFT) on the sensor data and generates a visualization of the frequency domain.

## Installation

Clone the repository: 
`git clone https://github.com/Ayman-tron/capstone.git`

Install the required packages:
`pip install numpy pandas scipy matplotlib scikit-learn firebase-admin`


## Usage

Each script is designed to be used in a pipeline, with `convertToTrainingData.py` first processing the raw data into a suitable format for machine learning, followed by `fault_detect.py` training the k-NN model and predicting the health status of the flange.

Use `plotEuclidean.py` and `plotFFT.py` to generate visualizations of the Euclidean distance between the sensor data points and the frequency domain of the sensor data, respectively.
