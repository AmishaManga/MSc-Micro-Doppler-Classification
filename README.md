# Micro-Doppler Classification of Humans and Animals Using FMCW Radar

Presented by: Amisha Manga

Prepared for: Department of Electrical Engineering, University of Cape Town in fulfilment of the academic requirements for a Master of Science Degree in Electrical Engineering with Specialization in Radar

Date: August 2024

## Documentation
* Some background information on how the Radar
* Radar Calculations
* Data Format

## Results
* Contains all the parameter optimizations and results obtained from the training the following classifiers:
    * Support Vector Machine
    * k-Nearest Neighbor
    * Random Forest
    * Convolutional Neural Network

* There are 3 different class label configurations:
    * Multi-Class: Human vs Animal Species
    * Multi-Class: Animal Species Only
    * Binary: Human vs Animal

* There are 2 types of datasets:
    * Full Dataset
    * Principle Component Analysis reduced data

## Python Source Code
* Contains the parameter optimisation scripts for the kNN, SVM and RF classifiers.
* Contains the PCA code
* Classifier code for kNN, SVM and RF
* Two stage classifiers
* Tensorflow CNN's
* Python data logging

## Matlab Source Code
* LCR Data Capture and Logging
* JeroMq (.jar file) for ZMQ integration into matlab
* Signal Processing
    * CFAR Algorithm
    * STFT Function
    * Spectrogram Generation
    * Manual Target Tracking
    * Range Doppler Maps
    * Alpha Filters

## Spectrograms
* Some interesting spectrograms generated from the research
    

