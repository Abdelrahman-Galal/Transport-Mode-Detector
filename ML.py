'''
This script is used for Bulidng/training/testing a ML model
Input: .csv files contains GPS data segments from prevoius scripts with the same order {Bus,Bike,Car,MobilityLog}
>> python ML.py Bus_Segs.csv Bike_Segs.csv Car_Segs.csv Mobility_Segs.csv Mobility_labeled_points.csv
====

Output: .csv file contain cleaned and segmented data
'''

#Import required libraries
from datetime import datetime
from datetime import timedelta
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from ast import literal_eval
from sklearn.neighbors import KNeighborsClassifier
import sys
import os

bus_file = sys.argv[1]
bike_file = sys.argv[2]
car_file = sys.argv[3]
mobility_file = sys.argv[4]
output_file = sys.argv[5]

max_accuracy = 0
max_model = "Initital"
def best_model(model1,acc1,model2,acc2):
    if acc1 > acc2 :
        max_accuracy = acc1
        max_model = model1
    else:
        max_accuracy = acc2
        max_model = model2
    return (max_model,max_accuracy)

#Read Input .csv(s)
print("Reading the input files")
df_bus = pd.read_csv(bus_file,float_precision='round_trip')
df_bike = pd.read_csv(bike_file,float_precision='round_trip')
df_car = pd.read_csv(car_file,float_precision='round_trip')
df_mobility = pd.read_csv(mobility_file,float_precision='round_trip')
df_mobility.points = df_mobility.points.apply(literal_eval)


#Choose feature columns and label column
df_bus_ML = df_bus[['average_speed','average_acceleration','Trans_mode']]
df_bike_ML = df_bike[['average_speed','average_acceleration','Trans_mode']]
df_car_ML = df_car[['average_speed','average_acceleration','Trans_mode']]
df_mobility_ML = df_mobility[['average_speed','average_acceleration']]
df_ML = df_bike_ML.append([df_bus_ML,df_car_ML],ignore_index=True)

#Create Balanced dataset
X_train,X_test,y_train,y_test = train_test_split(df_ML[['average_speed','average_acceleration']],df_ML.Trans_mode\
                                                 ,test_size=.3,random_state =42 )
X_test_speed = np.array(X_test['average_speed']).reshape(-1, 1)
X_test_acceleration = np.array(X_test['average_acceleration']).reshape(-1, 1)                                                 
df_ML_remaining = df_ML[~ df_ML.index.isin(y_test.index) ]
df_bike_ML_balanced = df_ML_remaining[ df_ML_remaining['Trans_mode'] == "Bike"].sample(n=24000,random_state=42)
df_bus_ML_balanced = df_ML_remaining[ df_ML_remaining['Trans_mode'] == "Bus"].sample(n=23730,random_state=42)
df_car_ML_balanced = df_ML_remaining[ df_ML_remaining['Trans_mode'] == "Car"].sample(n=23,random_state=5)
df_ML_balanced = df_bike_ML_balanced.append([df_bus_ML_balanced,df_car_ML_balanced])

X_train_balanced = df_ML_balanced[['average_speed','average_acceleration']] 
X_test_balanced = X_test
y_train_balanced = df_ML_balanced.Trans_mode
y_test_balanced = y_test

X_train_speed = np.array(X_train_balanced['average_speed']).reshape(-1, 1)
X_train_acceleration = np.array(X_train_balanced['average_acceleration']).reshape(-1, 1)

#KNN Speed as a feature
print("KNN model Speed")
k1_model = KNeighborsClassifier(n_neighbors=1)
k1_model = k1_model.fit(X_train_speed, y_train_balanced)
k1_y_pred = k1_model.predict(X_test_speed)
acc = accuracy_score(y_test_balanced,k1_y_pred)
max_model,max_accuracy = best_model(k1_model,acc,max_model,max_accuracy) 


k3_model = KNeighborsClassifier(n_neighbors=3)
k3_model = k3_model.fit(X_train_speed, y_train_balanced)
k3_y_pred = k3_model.predict(X_test_speed)
acc = accuracy_score(y_test_balanced,k3_y_pred)
max_model,max_accuracy = best_model(k3_model,acc,max_model,max_accuracy) 


k5_model = KNeighborsClassifier(n_neighbors=5)
k5_model = k5_model.fit(X_train_speed, y_train_balanced)
k5_y_pred = k5_model.predict(X_test_speed)
acc = accuracy_score(y_test_balanced,k5_y_pred)
max_model,max_accuracy = best_model(k5_model,acc,max_model,max_accuracy) 



k7_model = KNeighborsClassifier(n_neighbors=7)
k7_model = k7_model.fit(X_train_speed, y_train_balanced)
k7_y_pred = k7_model.predict(X_test_speed)
acc = accuracy_score(y_test_balanced,k7_y_pred)
max_model,max_accuracy = best_model(k7_model,acc,max_model,max_accuracy) 


k9_model = KNeighborsClassifier(n_neighbors=9)
k9_model = k9_model.fit(X_train_speed, y_train_balanced)
k9_y_pred = k9_model.predict(X_test_speed)
acc = accuracy_score(y_test_balanced,k9_y_pred)
max_model,max_accuracy = best_model(k7_model,acc,max_model,max_accuracy) 



#KNN Acceleration as a feature
print("KNN model Acceleration")
k1_model = KNeighborsClassifier(n_neighbors=1)
k1_model = k1_model.fit(X_train_acceleration, y_train_balanced)
k1_y_pred = k1_model.predict(X_test_acceleration)
acc = accuracy_score(y_test_balanced,k1_y_pred)
max_model,max_accuracy = best_model(k1_model,acc,max_model,max_accuracy)

k3_model = KNeighborsClassifier(n_neighbors=3)
k3_model = k3_model.fit(X_train_acceleration, y_train_balanced)
k3_y_pred = k3_model.predict(X_test_acceleration)
acc = accuracy_score(y_test_balanced,k3_y_pred)
max_model,max_accuracy = best_model(k3_model,acc,max_model,max_accuracy)

k5_model = KNeighborsClassifier(n_neighbors=5)
k5_model = k5_model.fit(X_train_acceleration, y_train_balanced)
k5_y_pred = k5_model.predict(X_test_acceleration)
acc = accuracy_score(y_test_balanced,k5_y_pred)
max_model,max_accuracy = best_model(k5_model,acc,max_model,max_accuracy)

k7_model = KNeighborsClassifier(n_neighbors=7)
k7_model = k7_model.fit(X_train_acceleration, y_train_balanced)
k7_y_pred = k7_model.predict(X_test_acceleration)
acc = accuracy_score(y_test_balanced,k7_y_pred)
max_model,max_accuracy = best_model(k7_model,acc,max_model,max_accuracy)

k9_model = KNeighborsClassifier(n_neighbors=9)
k9_model = k9_model.fit(X_train_acceleration, y_train_balanced)
k9_y_pred = k9_model.predict(X_test_acceleration)
acc = accuracy_score(y_test_balanced,k9_y_pred)
max_model,max_accuracy = best_model(k9_model,acc,max_model,max_accuracy)

#KNN Speed & Acceleration as a feature
print("KNN model Speed & Acceleration")
k1_model = KNeighborsClassifier(n_neighbors=1)
k1_model = k1_model.fit(X_train_balanced, y_train_balanced)
k1_y_pred = k1_model.predict(X_test_balanced)
acc = accuracy_score(y_test_balanced,k1_y_pred)
max_model,max_accuracy = best_model(k1_model,acc,max_model,max_accuracy)

k3_model = KNeighborsClassifier(n_neighbors=3)
k3_model = k3_model.fit(X_train_balanced, y_train_balanced)
k3_y_pred = k3_model.predict(X_test_balanced)
acc = accuracy_score(y_test_balanced,k3_y_pred)
max_model,max_accuracy = best_model(k3_model,acc,max_model,max_accuracy)


k5_model = KNeighborsClassifier(n_neighbors=5)
k5_model = k5_model.fit(X_train_balanced, y_train_balanced)
k5_y_pred = k5_model.predict(X_test_balanced)
acc = accuracy_score(y_test_balanced,k5_y_pred)


k7_model = KNeighborsClassifier(n_neighbors=7)
k7_model = k7_model.fit(X_train_balanced, y_train_balanced)
k7_y_pred = k7_model.predict(X_test_balanced)
acc = accuracy_score(y_test_balanced,k7_y_pred)


k9_model = KNeighborsClassifier(n_neighbors=9)
k9_model = k9_model.fit(X_train_balanced, y_train_balanced)
k9_y_pred = k9_model.predict(X_test_balanced)
acc = accuracy_score(y_test_balanced,k9_y_pred)
max_model,max_accuracy = best_model(k9_model,acc,max_model,max_accuracy)


#DT Speed as a feature
print("DT model Speed")
Dt = tree.DecisionTreeClassifier(random_state=42).fit(X_train_speed,y_train_balanced) 
DT_predection = Dt.predict(X_test_speed)
acc = accuracy_score(y_test_balanced,DT_predection)
max_model,max_accuracy = best_model(Dt,acc,max_model,max_accuracy)

#DT Acceleration as a feature
print("DT model Acceleration")
Dt = tree.DecisionTreeClassifier(random_state=42).fit(X_train_acceleration,y_train_balanced) 
DT_predection = Dt.predict(X_test_acceleration)
acc = accuracy_score(y_test_balanced,DT_predection)
max_model,max_accuracy = best_model(Dt,acc,max_model,max_accuracy)

#DT Speed & Acceleration as a feature
print("DT model Speed & Acceleration")
Dt = tree.DecisionTreeClassifier(random_state=42).fit(X_train_balanced, y_train_balanced) 
DT_predection = Dt.predict(X_test_balanced)
acc = accuracy_score(y_test_balanced,DT_predection)
max_model,max_accuracy = best_model(Dt,acc,max_model,max_accuracy)


#RF Speed as a feature
print("RF model Speed")
FT = RandomForestClassifier(n_estimators = 5 , random_state=42).fit(X_train_speed,y_train_balanced)
FT_predection = FT.predict(X_test_speed)
acc = accuracy_score(y_test_balanced,FT_predection)
max_model,max_accuracy = best_model(FT,acc,max_model,max_accuracy)

FT = RandomForestClassifier(n_estimators = 10 , random_state=42).fit(X_train_speed,y_train_balanced)
FT_predection = FT.predict(X_test_speed)
acc = accuracy_score(y_test_balanced,FT_predection)
max_model,max_accuracy = best_model(FT,acc,max_model,max_accuracy)

FT = RandomForestClassifier(n_estimators = 20 , random_state=42).fit(X_train_speed,y_train_balanced)
FT_predection = FT.predict(X_test_speed)
acc = accuracy_score(y_test_balanced,FT_predection)
max_model,max_accuracy = best_model(FT,acc,max_model,max_accuracy)

FT = RandomForestClassifier(n_estimators = 30 , random_state=42).fit(X_train_speed,y_train_balanced)
FT_predection = FT.predict(X_test_speed)
acc = accuracy_score(y_test_balanced,FT_predection)
max_model,max_accuracy = best_model(FT,acc,max_model,max_accuracy)

FT = RandomForestClassifier(n_estimators = 40 , random_state=42).fit(X_train_speed,y_train_balanced)
FT_predection = FT.predict(X_test_speed)
acc = accuracy_score(y_test_balanced,FT_predection)
max_model,max_accuracy = best_model(FT,acc,max_model,max_accuracy)


#RF Acceleration as a feature
print("RF model Acceleration")
FT = RandomForestClassifier(n_estimators = 5 , random_state=42).fit(X_train_acceleration,y_train_balanced)
FT_predection = FT.predict(X_test_acceleration)
acc = accuracy_score(y_test_balanced,FT_predection)
max_model,max_accuracy = best_model(FT,acc,max_model,max_accuracy)

FT = RandomForestClassifier(n_estimators = 10 , random_state=42).fit(X_train_acceleration,y_train_balanced)
FT_predection = FT.predict(X_test_acceleration)
acc = accuracy_score(y_test_balanced,FT_predection)
max_model,max_accuracy = best_model(FT,acc,max_model,max_accuracy)

FT = RandomForestClassifier(n_estimators = 20 , random_state=42).fit(X_train_acceleration,y_train_balanced)
FT_predection = FT.predict(X_test_acceleration)
acc = accuracy_score(y_test_balanced,FT_predection)
max_model,max_accuracy = best_model(FT,acc,max_model,max_accuracy)

FT = RandomForestClassifier(n_estimators = 30 , random_state=42).fit(X_train_acceleration,y_train_balanced)
FT_predection = FT.predict(X_test_acceleration)
acc = accuracy_score(y_test_balanced,FT_predection)
max_model,max_accuracy = best_model(FT,acc,max_model,max_accuracy)

FT = RandomForestClassifier(n_estimators = 40 , random_state=42).fit(X_train_acceleration,y_train_balanced)
FT_predection = FT.predict(X_test_acceleration)
acc = accuracy_score(y_test_balanced,FT_predection)
max_model,max_accuracy = best_model(FT,acc,max_model,max_accuracy)

#RF Speed & Acceleration as a feature
print("RF model Speed & Acceleration")
FT5 = RandomForestClassifier(n_estimators = 5 , random_state=42).fit(X_train_balanced, y_train_balanced)
FT5_predection = FT5.predict(X_test_balanced)
acc = accuracy_score(y_test_balanced,FT5_predection)
max_model,max_accuracy = best_model(FT5,acc,max_model,max_accuracy)

FT10 = RandomForestClassifier(n_estimators = 10 , random_state=42).fit(X_train_balanced, y_train_balanced)
FT10_predection = FT10.predict(X_test_balanced)
acc = accuracy_score(y_test_balanced,FT10_predection)
max_model,max_accuracy = best_model(FT10,acc,max_model,max_accuracy)

FT20 = RandomForestClassifier(n_estimators = 20 , random_state=42).fit(X_train_balanced, y_train_balanced)
FT20_predection = FT20.predict(X_test_balanced)
acc = accuracy_score(y_test_balanced,FT20_predection)
max_model,max_accuracy = best_model(FT20,acc,max_model,max_accuracy)

FT30 = RandomForestClassifier(n_estimators = 30 , random_state=42).fit(X_train_balanced, y_train_balanced)
FT30_predection = FT30.predict(X_test_balanced)
acc = accuracy_score(y_test_balanced,FT30_predection)
max_model,max_accuracy = best_model(FT30,acc,max_model,max_accuracy)

FT40 = RandomForestClassifier(n_estimators = 40 , random_state=42).fit(X_train_balanced, y_train_balanced)
FT40_predection = FT40.predict(X_test_balanced)
acc = accuracy_score(y_test_balanced,FT40_predection)
max_model,max_accuracy = best_model(FT40,acc,max_model,max_accuracy)

print(f"Model with max acc so far is {max_model} with accuracy of {max_accuracy}")

print(confusion_matrix(y_test_balanced,max_model.predict(X_test_balanced)))
matrix = classification_report(y_test_balanced,FT10_predection,labels=max_model.classes_,digits=3)
print(matrix)

mobility_predict = max_model.predict(df_mobility_ML)
df_mobility['Trans_mode'] = mobility_predict
print(df_mobility['Trans_mode'].value_counts(normalize=True))

mobility = df_mobility.values.tolist()
predict_points_labeled = []
for i in range(len(mobility)):
    for point in mobility[i][13]:
        tmp = []
        key = i
        latitude,longitude = point
        total_points = mobility[i][8]
        trans_mode = mobility[i][14]
        tmp.extend([key,latitude,longitude,total_points,trans_mode])
        predict_points_labeled.append(tmp)

df_predict_points_labeled = pd.DataFrame(predict_points_labeled,columns=["key","latitude","longitude",\
                                                                  "total_points","Trans_mode"])
print(f"Saving output to {output_file}")                                                                  
df_predict_points_labeled.to_csv(path_or_buf=output_file,index=False)
print("Completed...")

