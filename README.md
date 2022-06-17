# Transport-Mode-Detector
This is framework used to conduct a master's thesis titled *Detecting Public Transport Mode in The City of Tartu Using
Smartphone-Based GPS Data and Machine Learning Methods*.

The general framework includes the below steps:
1. Data cleaning (missing values, duplicate values, and outliers)
2. Data wrangling and features engineering (calculate haversin distance between GPS points,speed and acceleration)
3. Data segmentation (aggergate GPS points to trips and segments)
4. Applying supervised machine learning models and choose the best model based on accuracy,recall and precision
5. Using the best model to predict transport mode of unlabeled data

Detailed info about data,methods and results can be found [here](https://dspace.ut.ee/handle/10062/72816).
