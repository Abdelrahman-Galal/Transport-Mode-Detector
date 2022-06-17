'''
This script is used for data cleaning and segmentation of Car GPS data coming of Komoot mobile app
Input: .csv file contains GPS data in the below format
====
user	timestamp	latitude	longitude
2	2021-03-08T15:12:58.880Z	58.368755	26.722853
2	2021-03-08T15:13:54.177Z	58.368576	26.722887
2	2021-03-08T15:13:57.197Z	58.368458	26.722882
2	2021-03-08T15:14:10.180Z	58.368374	26.722977
2	2021-03-08T15:14:12.171Z	58.368411	26.723159

====
Output: .csv file contain cleaned and segmented data
'''

#Import required libraries
from datetime import datetime
from datetime import timedelta
from math import radians,cos,sin,asin,sqrt
import numpy as np
import pandas as pd
import sys
import os

input_file = sys.argv[1]
output_file = sys.argv[2]

#Read Input .csv
print("Reading the input file")
car_gps_df = pd.read_csv(input_file,float_precision='round_trip')

#Format the time
FMT = '%H:%M:%S'
car_gps_df['coord_date'] = car_gps_df['timestamp'].str[0:10]
car_gps_df['coord_time'] = car_gps_df['timestamp'].str[11:19]
car_gps_df['coord_time'] = pd.to_datetime(car_gps_df['coord_time'],format=FMT)
car_gps_df = car_gps_df.sort_values(by=['user','coord_date','coord_time'],ascending=True)\
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
car_gps_df['date_start'] = car_gps_df['coord_date']
car_gps_df['date_end'] = car_gps_df['coord_date'].iloc[1:].reset_index(drop=True)
car_gps_df['time_start'] = car_gps_df['coord_time']
car_gps_df['time_end'] = car_gps_df['coord_time'].iloc[1:].reset_index(drop=True)
car_gps_df['latitude_start'] = car_gps_df['latitude']
car_gps_df['longitude_start'] = car_gps_df['longitude']
car_gps_df['latitude_end'] = car_gps_df['latitude'].iloc[1:].reset_index(drop=True)
car_gps_df['longitude_end'] = car_gps_df['longitude'].iloc[1:].reset_index(drop=True)
car_gps_df['User'] = car_gps_df['user']
car_gps_df['user_check'] = car_gps_df['user'].iloc[1:].reset_index(drop=True)
car_gps_df = car_gps_df.\
        drop(['timestamp','coord_date','coord_time','latitude','longitude','user'],axis='columns')
car_gps_df = car_gps_df.drop([len(car_gps_df) - 1], axis='index')
car_gps_df.rename(columns={"User":"user"},inplace=True)

#Keep only rows that has the same start and end data and belongs to the same user
car_gps_df = car_gps_df[(car_gps_df.date_start == car_gps_df.date_end) \
                          & (car_gps_df.user == car_gps_df.user_check)].reset_index(drop=True)

#Calculate time delta in seconds between each congestive points 
print("Time delta calculation")                        
car_gps_df['time_diff'] = car_gps_df['time_end'] - car_gps_df['time_start']
time_diff = car_gps_df['time_diff'].to_list()
time_diff_seconds = [timedelta.total_seconds(diff) for diff in time_diff]
car_gps_df['time_diff_seconds'] = time_diff_seconds     

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
longitude_start = car_gps_df['longitude_start'].to_list()
latitude_start = car_gps_df['latitude_start'].to_list()
longitude_end = car_gps_df['longitude_end'].to_list()
latitude_end = car_gps_df['latitude_end'].to_list()
car_gps_df['distance' ]=[haversine(lon1,lat1,lon2,lat2) \
           for lon1,lat1,lon2,lat2 in zip(longitude_start,latitude_start,longitude_end,latitude_end)]  


#Drop points with zero time delta
car_gps_df   = car_gps_df[ car_gps_df['time_diff_seconds'] != 0 ].reset_index(drop=True)

#Calculate speed (meters/second)
car_gps_df['speed'] = car_gps_df['distance']/car_gps_df['time_diff_seconds']

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
users = car_gps_df['user'].unique()
user_group = car_gps_df.groupby(['user'])
#List that contain all segment
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
            #if time delta between two points is more than 2 min or distance is larger than Car average speed (40 meters/second) * time delta
            # then end the current segment and open new one
            if row[11] > 120 or row[12] > row[11]*40:
                time_end = row[2]
                latitude_end = row[4]    
                longitude_end = row[5]
                average_speed = sum(speed)/len(speed) if len(speed) > 0 else 0
                average_acceleration = Average_Acceleration(speed,time) if len(speed) > 2 else 0
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
            #Add GPS points to the segment    
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
df_segs = df_segs[df_segs['point_count'] > 0 ]  

#Add transport mode as {Car}
df_segs['Trans_mode'] = 'Car'

#Save the the output to .csv file
print(f"Saving output to {output_file}")
df_segs.to_csv(path_or_buf=output_file,index=False)   
print("Completed...")      