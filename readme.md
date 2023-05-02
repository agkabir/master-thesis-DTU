# Rainfall forecasting from radar inages using Deep Learning

This repository contains the code for data pre-processing and different model architectures outlined in the thesis paper. The code in written in python.

## Table of contents

- [Overview](#overview)
  - [Requirements](#requirements)
  - [Steps to train model for whole Denmark](#steps-to-train-model-for-whole-denmark)
  - [Steps to test a trained model for whole Denmark:](#steps-to-test-a-trained-model-for-whole-denmark:)
  - [Screenshot](#screenshot)
  - [Links](#links)
- [My process](#my-process)
  - [Built with](#built-with)
  - [What I learned](#what-i-learned)
  - [Useful resources](#useful-resources)
- [Author](#author)

## Overview

### Requirements

    - Python 3.6
    - TensorFlow 2.3.0
    - h5py 2.10.0
    - pandas 1.3.4
    - numpy 1.21.2

### Steps to train model for whole Denmark:

1. Process the radar observations by running "data_proessing_whole_denmark.py" with the necessary informations for example: supplying data path, processed file path, periods (2016-12,2017-01,......) etc. It will generate processed files as "2016-12.array" and "2016-12.csv", "2017-01.array" and "2017-01.csv".....
2. Put the processed .array files under the folder called "Data" and .csv files under the folder called csv_files.
3. For whole Denmark a file named "mask.array" is needed which was formed by setting 0 to pixel values out of radar coverage area and 1 to pixel values inside of radar coverage.
4. Train the model by running "train.py". At the end of the training the trained model will be saved as 'dmi_checkpoints.h5' and indexes of test dataset will be generated.

### Steps to test a trained model for whole Denmark:

1. Run the "test.py" simply by specifying the path of test dataset index file.

## Steps to train a model for cropped region:

1. Process the radar observations and radar nowcasts by running "data_proessing_cropped_observation.py" and "data_proessing_cropped_nowcast.py" with the necessary informations for example: supplying data path, processed file path, periods (2016-12,2017-01,......) etc. It will generate processed files as "2016-12.array" and "2016-12.csv", "2017-01.array" and "2017-01.csv".....
2. Put the processed .array files under the folder called "Data/observation" & .csv files under the folder called csv_files for radar observation and put the processed .array files under the folder called "Data/nowcast" for radar nowcasts.
3. Finally, train the model by running "train.py".
