This repository contains the code for <b> Fuzzy Weighted Progressive Time Contortion (FWPTC) </b>, a method for data augmentaton of timeseries for regression. This is an extension of DTW algorithm using fuzzy sets. The method is applied for <b> EEG based reaction time prediction</b> using deep learning (<b>EEGNet</b>). Reaction time prediction helps to understand brain dynamics and behavior relationship in a <b>driver fatigue prediction</b> task.


## Code
The code is divided as follows: 
* The [power_data_augment_code.py](https://github.com/islitik123/TFS936.R1/blob/master/Final_code_submit/power_data_augment_code.py) file contains the method that generates the random weights (Average Selected) with a function that does the actual augmentation for a given training set of time series.
* The [distance](https://github.com/islitik123/TFS936.R1/tree/master/Final_code_submit/distances/dtw) folder contains the Fuzzy based DTW distance in Cython . 
* The [knn.py](https://github.com/islitik123/TFS936.R1/blob/master/Final_code_submit/knn.py) file contains the K nearest neighbor algorithm which is mainly used when computing the weights for the data augmentation technique.
* The [dba.py](https://github.com/islitik123/TFS936.R1/blob/master/Final_code_submit/dba.py) file contains the DBA algorithm.  
* The [utils](https://github.com/islitik123/TFS936.R1/tree/master/Final_code_submit/utils) folder contains the necessary functions to read the datasets and visualize the plots.  
* The [eegnet_for_power.py](https://github.com/islitik123/TFS936.R1/blob/master/Final_code_submit/eegnet_for_power.py) file contains the keras and tensorflow code to define the architecture of eegnet and train the deep learning model for unaugmented data.
* The [eegnet_for_aug_power.py](https://github.com/islitik123/TFS936.R1/blob/master/Final_code_submit/eegnet_for_aug_power.py) file contains the keras and tensorflow code to define the architecture of eegnet and train the deep learning model for FWPTC augmented data. 
