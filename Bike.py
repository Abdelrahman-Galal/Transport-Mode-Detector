'''
This script is used for data cleaning and segmentation of Tartu shared bike system GPS data
Input: .csv file contains GPS data in the below format
====
route_code	cyclenumber	latitude	longitude	coord_date	coord_time	userID_new
1.55956E+12	2421	58.379825	26.72064	03/06/2019	12:48:31+00	8693
1.55956E+12	2421	58.379865	26.720405	03/06/2019	12:48:36+00	8693
1.55956E+12	2421	58.379875	26.71998333	03/06/2019	12:48:41+00	8693
====
Output: .csv file contain cleaned and segmented data
'''

#Import required libraries
from datetime import datetime
from datetime import timedelta
from math import radians,cos,sin,asin,sqrt
import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

input_file = sys.argv[1]
output_file = sys.argv[2]

#Read Input .csv
print("Reading the input file")
bike_gps_df = pd.read_csv(input_file,float_precision='round_trip')

#Drop NA rows
bike_gps_df = bike_gps_df.dropna().reset_index(drop=True)

#Format the time
FMT = '%H:%M:%S'
bike_gps_df['coord_time'] = bike_gps_df['coord_time'].str[0:8]
bike_gps_df['coord_time'] = pd.to_datetime(bike_gps_df['coord_time'],format=FMT)

#Sort the movement 
bike_gps_df = bike_gps_df.sort_values(by=['userID_new','coord_date','coord_time'],ascending=True)\
                                                                                    .reset_index(drop=True)

#Wrangle the data to be in the form below
#   Column           Dtype         
#---  ------           -----         
# 0   date_start       object        
# 1   date_end         object        
# 2   time_start       datetime64[ns]
# 3   time_end         datetime64[ns]
# 4   latitude_start   float64       
# 5   longitude_start  float64       
# 6   latitude_end     float64       
# 7   longitude_end    float64       
# 8   user             int64         
# 9   user_check       float64 
print("Data cleaning and wrangling")                                                                                  
bike_gps_df['date_start'] = bike_gps_df['coord_date']
bike_gps_df['date_end'] = bike_gps_df['coord_date'].iloc[1:].reset_index(drop=True)
bike_gps_df['time_start'] = bike_gps_df['coord_time']
bike_gps_df['time_end'] = bike_gps_df['coord_time'].iloc[1:].reset_index(drop=True)
bike_gps_df['latitude_start'] = bike_gps_df['latitude']
bike_gps_df['longitude_start'] = bike_gps_df['longitude']
bike_gps_df['latitude_end'] = bike_gps_df['latitude'].iloc[1:].reset_index(drop=True)
bike_gps_df['longitude_end'] = bike_gps_df['longitude'].iloc[1:].reset_index(drop=True)
bike_gps_df['user'] = bike_gps_df['userID_new']
bike_gps_df['user_check'] = bike_gps_df['userID_new'].iloc[1:].reset_index(drop=True)
bike_gps_df = bike_gps_df.\
        drop(['route_code','cyclenumber','coord_date','coord_time','latitude','longitude','userID_new'],axis='columns')
bike_gps_df = bike_gps_df.drop([len(bike_gps_df) - 1], axis='index')                                                                                   
                                                                                    
#Keep only rows that has the same start and end data and belongs to the same user
bike_gps_df = bike_gps_df[(bike_gps_df.date_start == bike_gps_df.date_end) \
                          & (bike_gps_df.user == bike_gps_df.user_check)].reset_index(drop=True)  


#Calculate time delta in seconds between each congestive points
print("Time delta calculation")
bike_gps_df['time_diff'] = bike_gps_df['time_end']- bike_gps_df['time_start']
time_diff = bike_gps_df['time_diff'].to_list()
time_diff_seconds = [timedelta.total_seconds(diff) for diff in time_diff]
bike_gps_df['time_diff_seconds'] = time_diff_seconds


#Function to be used in distance calculations (meters) usinf haversine formula
def haversine(lon1,lat1,lon2,lat2):
    lon1,lat1,lon2,lat2 = map(radians,[lon1,lat1,lon2,lat2])
    dlon = lon2-lon1
    dlat = lat2-lat1
    a = sin(dlat/2)**2+cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c= 2*asin(sqrt(a))
    r=6371
    return (c*r)*1000 

#Calculate distance in meters between each congestive points
print("Distance delta calculation")
longitude_start = bike_gps_df['longitude_start'].to_list()
latitude_start = bike_gps_df['latitude_start'].to_list()
longitude_end = bike_gps_df['longitude_end'].to_list()
latitude_end = bike_gps_df['latitude_end'].to_list()
bike_gps_df['distance' ]=[haversine(lon1,lat1,lon2,lat2) \
           for lon1,lat1,lon2,lat2 in zip(longitude_start,latitude_start,longitude_end,latitude_end)]
           
#Drop points with zero time delta
bike_gps_df   = bike_gps_df[ bike_gps_df['time_diff_seconds'] != 0 ].reset_index(drop=True)
    
#Calculate speed (meters/second)
bike_gps_df['speed'] = bike_gps_df['distance']/bike_gps_df['time_diff_seconds'] 

#Function to be used in Acceleration calculation 
def Average_Acceleration(speed,time):
    acceleration = []
    for i in range(1,len(speed)):
        acceleration.append((speed[i]-speed[i-1])/time[i-1])
    return sum(acceleration)/len(acceleration) 
    
#Segmentation process
# For each user in the dataset , loop over the days in the track
#------------------------
print("Segmentation process started")
users = bike_gps_df['user'].unique()
user_group = bike_gps_df.groupby(['user'])
#List that contain all segments
segs = []
for user in users:
    df = user_group.get_group(user)
    days = df['date_start'].unique()
    days_group = df.groupby(['date_start'])
    for day in days:
        df_user_day = days_group.get_group(day)
        df_user_day_list = df_user_day.values.tolist()
        seg = []
        speed = []
        time = []   
        count = 0
        total_time = 0
        total_distance = 0
        time_start = df_user_day_list[0][2]
        latitude_start = df_user_day_list[0][4]         
        longitude_start   = df_user_day_list[0][5]
        seg.append(user)
        for index,row in enumerate(df_user_day_list):
            # if the time delta is larger than 1 min or distance is larger than average bike speed (15 meters/second) * time delta
            # then end the current segment and open new one
            if row[11] > 60 or row[12] > row[11]*15:
                time_end = row[2]
                latitude_end = row[4]    
                longitude_end = row[5]
                average_speed = sum(speed)/len(speed) if len(speed) > 0 else 0
                average_acceleration = Average_Acceleration(speed,time) if len(speed) > 2 else 0
                count = count+1
                seg.extend([day,time_start,time_end,latitude_start,longitude_start,\
                            latitude_end,longitude_end,\
                            count,total_time,total_distance,\
                            average_speed,average_acceleration])
                # Finish current segment
                segs.append(seg)
                #Start new segment
                seg = []
                seg.append(user)
                speed = []
                time = []
                count = 0
                total_time = 0
                total_distance = 0
                try:
                    time_start = df_user_day_list[index+1][2]
                    latitude_start = df_user_day_list[index+1][4]         
                    longitude_start   = df_user_day_list[index+1][5]
                except:
                    pass
            #Last segment in the dataframe
            elif index == len(df_user_day_list)-1:
                time_end = row[3]
                latitude_end = row[6]    
                longitude_end = row[7] 
                count = count+1
                total_time = total_time+row[11]
                total_distance = total_distance+row[12]
                speed.append(row[13])
                time.append(row[11])
                average_speed = sum(speed)/len(speed) if len(speed) > 0 else 0
                average_acceleration = Average_Acceleration(speed,time) if len(speed) > 2 else 0
                seg.extend([day,time_start,time_end,latitude_start,longitude_start,\
                                     latitude_end,longitude_end,\
                                     count,total_time,total_distance,\
                                     average_speed,average_acceleration])
                segs.append(seg)
            #Add points to the segment    
            else:
                count = count + 1
                total_time = total_time + row[11]
                total_distance = total_distance+row[12]            
                speed.append(row[13])   
                time.append(row[11])
                
# Convert the list to dataframe
df_segs = pd.DataFrame(segs,columns=['user','date','time_start','time_end','latitude_start','longitude_start',\
                                     'latitude_end','longitude_end',\
                                     'point_count','total_time','total_distance',\
                                     'average_speed','average_acceleration'])

print("Segmentation process ended")                                    
#Drop short segments
df_segs = df_segs[df_segs.total_distance > 60 ]
df_segs = df_segs[df_segs.point_count > 20 ] 

#Add transport mode as {Bike}
df_segs['Trans_mode'] = 'Bike'

#Save the the output to .csv file
print(f"Saving output to {output_file}")
df_segs.to_csv(path_or_buf= output_file,index=False) 
print("Completed...")
    