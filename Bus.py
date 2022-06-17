'''
This script is used for data cleaning and segmentation of Bus GPS data coming of Tartu City public transport system for bus trips (trip-based)
Each day in one file name 1,2,.... 
Input: .csv file contains GPS data in the below format
====
	id	stopTimeUpdates	trip.routeId	trip.directionId	trip.tripId	trip.startTime	trip.startDate	currentStatus	stopId	currentStopSequence	timestamp	position.bearing	position.latitude	position.speed	position.longitude	vehicle.id	vehicle.label
0	15405	[{'stopSequence': 1, 'arrival': {'delay': -9009, 'time': 1601970471}, 'stopId': '7820165-1', 'departure': {'delay': 0, 'time': 1601979480}}, {'stopSequence': 2, 'arrival': {'delay': 0, 'time': 1601979540}, 'stopId': '7820155-1', 'departure': {'delay': 0, 'time': 1601979540}}, {'stopSequence': 3, 'arrival': {'delay': 0, 'time': 1601979660}, 'stopId': '7820014-1', 'departure': {'delay': 0, 'time': 1601979660}}, {'stopSequence': 4, 'arrival': {'delay': 0, 'time': 1601979720}, 'stopId': '7820064-1', 'departure': {'delay': 0, 'time': 1601979720}}, {'stopSequence': 5, 'arrival': {'delay': 0, 'time': 1601979840}, 'stopId': '7820028-1', 'departure': {'delay': 0, 'time': 1601979840}}, {'stopSequence': 6, 'arrival': {'delay': 0, 'time': 1601979900}, 'stopId': '7820068-1', 'departure': {'delay': 0, 'time': 1601979900}}, {'stopSequence': 7, 'arrival': {'delay': 0, 'time': 1601980020}, 'stopId': '7820187-1', 'departure': {'delay': 0, 'time': 1601980020}}, {'stopSequence': 8, 'arrival': {'delay': 0, 'time': 1601980140}, 'stopId': '7820020-1', 'departure': {'delay': 0, 'time': 1601980140}}, {'stopSequence': 9, 'arrival': {'delay': 0, 'time': 1601980320}, 'stopId': '7820085-2', 'departure': {'delay': 0, 'time': 1601980380}}, {'stopSequence': 10, 'arrival': {'delay': 0, 'time': 1601980440}, 'stopId': '7820219-1', 'departure': {'delay': 0, 'time': 1601980440}}, {'stopSequence': 11, 'arrival': {'delay': 0, 'time': 1601980500}, 'stopId': '7820177-1', 'departure': {'delay': 0, 'time': 1601980500}}, {'stopSequence': 12, 'arrival': {'delay': 0, 'time': 1601980620}, 'stopId': '7820126-1', 'departure': {'delay': 0, 'time': 1601980620}}, {'stopSequence': 13, 'arrival': {'delay': 0, 'time': 1601980680}, 'stopId': '7820059-1', 'departure': {'delay': 0, 'time': 1601980680}}, {'stopSequence': 14, 'arrival': {'delay': 0, 'time': 1601980800}, 'stopId': '7820242-1', 'departure': {'delay': 0, 'time': 1601980800}}, {'stopSequence': 15, 'arrival': {'delay': 0, 'time': 1601980860}, 'stopId': '7820200-1', 'departure': {'delay': 0, 'time': 1601980860}}, {'stopSequence': 16, 'arrival': {'delay': 0, 'time': 1601980920}, 'stopId': '7820217-1', 'departure': {'delay': 0, 'time': 1601980920}}, {'stopSequence': 17, 'arrival': {'delay': 0, 'time': 1601980980}, 'stopId': '7820276-1', 'departure': {'delay': 0, 'time': 1601980980}}, {'stopSequence': 18, 'arrival': {'delay': 0, 'time': 1601981220}, 'stopId': '7820035-2', 'departure': {'delay': 0, 'time': 1601981220}}]	1	2	1011557	13:18:00	06/10/2020	IN_TRANSIT_TO	7820165-1	1	1602023391	230	58.379826	0	26.721577	15405	Live Pos
1	15405	[{'stopSequence': 1, 'arrival': {'delay': -9009, 'time': 1601970471}, 'stopId': '7820165-1', 'departure': {'delay': 0, 'time': 1601979480}}, {'stopSequence': 2, 'arrival': {'delay': 0, 'time': 1601979540}, 'stopId': '7820155-1', 'departure': {'delay': 0, 'time': 1601979540}}, {'stopSequence': 3, 'arrival': {'delay': 0, 'time': 1601979660}, 'stopId': '7820014-1', 'departure': {'delay': 0, 'time': 1601979660}}, {'stopSequence': 4, 'arrival': {'delay': 0, 'time': 1601979720}, 'stopId': '7820064-1', 'departure': {'delay': 0, 'time': 1601979720}}, {'stopSequence': 5, 'arrival': {'delay': 0, 'time': 1601979840}, 'stopId': '7820028-1', 'departure': {'delay': 0, 'time': 1601979840}}, {'stopSequence': 6, 'arrival': {'delay': 0, 'time': 1601979900}, 'stopId': '7820068-1', 'departure': {'delay': 0, 'time': 1601979900}}, {'stopSequence': 7, 'arrival': {'delay': 0, 'time': 1601980020}, 'stopId': '7820187-1', 'departure': {'delay': 0, 'time': 1601980020}}, {'stopSequence': 8, 'arrival': {'delay': 0, 'time': 1601980140}, 'stopId': '7820020-1', 'departure': {'delay': 0, 'time': 1601980140}}, {'stopSequence': 9, 'arrival': {'delay': 0, 'time': 1601980320}, 'stopId': '7820085-2', 'departure': {'delay': 0, 'time': 1601980380}}, {'stopSequence': 10, 'arrival': {'delay': 0, 'time': 1601980440}, 'stopId': '7820219-1', 'departure': {'delay': 0, 'time': 1601980440}}, {'stopSequence': 11, 'arrival': {'delay': 0, 'time': 1601980500}, 'stopId': '7820177-1', 'departure': {'delay': 0, 'time': 1601980500}}, {'stopSequence': 12, 'arrival': {'delay': 0, 'time': 1601980620}, 'stopId': '7820126-1', 'departure': {'delay': 0, 'time': 1601980620}}, {'stopSequence': 13, 'arrival': {'delay': 0, 'time': 1601980680}, 'stopId': '7820059-1', 'departure': {'delay': 0, 'time': 1601980680}}, {'stopSequence': 14, 'arrival': {'delay': 0, 'time': 1601980800}, 'stopId': '7820242-1', 'departure': {'delay': 0, 'time': 1601980800}}, {'stopSequence': 15, 'arrival': {'delay': 0, 'time': 1601980860}, 'stopId': '7820200-1', 'departure': {'delay': 0, 'time': 1601980860}}, {'stopSequence': 16, 'arrival': {'delay': 0, 'time': 1601980920}, 'stopId': '7820217-1', 'departure': {'delay': 0, 'time': 1601980920}}, {'stopSequence': 17, 'arrival': {'delay': 0, 'time': 1601980980}, 'stopId': '7820276-1', 'departure': {'delay': 0, 'time': 1601980980}}, {'stopSequence': 18, 'arrival': {'delay': 0, 'time': 1601981220}, 'stopId': '7820035-2', 'departure': {'delay': 0, 'time': 1601981220}}]	1	2	1011557	13:18:00	06/10/2020	IN_TRANSIT_TO	7820165-1	1	1602023380	230	58.379826	0	26.721577	15405	Live Pos
2	15405	[{'stopSequence': 1, 'arrival': {'delay': -9009, 'time': 1601970471}, 'stopId': '7820165-1', 'departure': {'delay': 0, 'time': 1601979480}}, {'stopSequence': 2, 'arrival': {'delay': 0, 'time': 1601979540}, 'stopId': '7820155-1', 'departure': {'delay': 0, 'time': 1601979540}}, {'stopSequence': 3, 'arrival': {'delay': 0, 'time': 1601979660}, 'stopId': '7820014-1', 'departure': {'delay': 0, 'time': 1601979660}}, {'stopSequence': 4, 'arrival': {'delay': 0, 'time': 1601979720}, 'stopId': '7820064-1', 'departure': {'delay': 0, 'time': 1601979720}}, {'stopSequence': 5, 'arrival': {'delay': 0, 'time': 1601979840}, 'stopId': '7820028-1', 'departure': {'delay': 0, 'time': 1601979840}}, {'stopSequence': 6, 'arrival': {'delay': 0, 'time': 1601979900}, 'stopId': '7820068-1', 'departure': {'delay': 0, 'time': 1601979900}}, {'stopSequence': 7, 'arrival': {'delay': 0, 'time': 1601980020}, 'stopId': '7820187-1', 'departure': {'delay': 0, 'time': 1601980020}}, {'stopSequence': 8, 'arrival': {'delay': 0, 'time': 1601980140}, 'stopId': '7820020-1', 'departure': {'delay': 0, 'time': 1601980140}}, {'stopSequence': 9, 'arrival': {'delay': 0, 'time': 1601980320}, 'stopId': '7820085-2', 'departure': {'delay': 0, 'time': 1601980380}}, {'stopSequence': 10, 'arrival': {'delay': 0, 'time': 1601980440}, 'stopId': '7820219-1', 'departure': {'delay': 0, 'time': 1601980440}}, {'stopSequence': 11, 'arrival': {'delay': 0, 'time': 1601980500}, 'stopId': '7820177-1', 'departure': {'delay': 0, 'time': 1601980500}}, {'stopSequence': 12, 'arrival': {'delay': 0, 'time': 1601980620}, 'stopId': '7820126-1', 'departure': {'delay': 0, 'time': 1601980620}}, {'stopSequence': 13, 'arrival': {'delay': 0, 'time': 1601980680}, 'stopId': '7820059-1', 'departure': {'delay': 0, 'time': 1601980680}}, {'stopSequence': 14, 'arrival': {'delay': 0, 'time': 1601980800}, 'stopId': '7820242-1', 'departure': {'delay': 0, 'time': 1601980800}}, {'stopSequence': 15, 'arrival': {'delay': 0, 'time': 1601980860}, 'stopId': '7820200-1', 'departure': {'delay': 0, 'time': 1601980860}}, {'stopSequence': 16, 'arrival': {'delay': 0, 'time': 1601980920}, 'stopId': '7820217-1', 'departure': {'delay': 0, 'time': 1601980920}}, {'stopSequence': 17, 'arrival': {'delay': 0, 'time': 1601980980}, 'stopId': '7820276-1', 'departure': {'delay': 0, 'time': 1601980980}}, {'stopSequence': 18, 'arrival': {'delay': 0, 'time': 1601981220}, 'stopId': '7820035-2', 'departure': {'delay': 0, 'time': 1601981220}}]	1	2	1011557	13:18:00	06/10/2020	IN_TRANSIT_TO	7820165-1	1	1602023370	230	58.379826	0	26.721577	15405	Live Pos
3	15405	[{'stopSequence': 1, 'arrival': {'delay': -9009, 'time': 1601970471}, 'stopId': '7820165-1', 'departure': {'delay': 0, 'time': 1601979480}}, {'stopSequence': 2, 'arrival': {'delay': 0, 'time': 1601979540}, 'stopId': '7820155-1', 'departure': {'delay': 0, 'time': 1601979540}}, {'stopSequence': 3, 'arrival': {'delay': 0, 'time': 1601979660}, 'stopId': '7820014-1', 'departure': {'delay': 0, 'time': 1601979660}}, {'stopSequence': 4, 'arrival': {'delay': 0, 'time': 1601979720}, 'stopId': '7820064-1', 'departure': {'delay': 0, 'time': 1601979720}}, {'stopSequence': 5, 'arrival': {'delay': 0, 'time': 1601979840}, 'stopId': '7820028-1', 'departure': {'delay': 0, 'time': 1601979840}}, {'stopSequence': 6, 'arrival': {'delay': 0, 'time': 1601979900}, 'stopId': '7820068-1', 'departure': {'delay': 0, 'time': 1601979900}}, {'stopSequence': 7, 'arrival': {'delay': 0, 'time': 1601980020}, 'stopId': '7820187-1', 'departure': {'delay': 0, 'time': 1601980020}}, {'stopSequence': 8, 'arrival': {'delay': 0, 'time': 1601980140}, 'stopId': '7820020-1', 'departure': {'delay': 0, 'time': 1601980140}}, {'stopSequence': 9, 'arrival': {'delay': 0, 'time': 1601980320}, 'stopId': '7820085-2', 'departure': {'delay': 0, 'time': 1601980380}}, {'stopSequence': 10, 'arrival': {'delay': 0, 'time': 1601980440}, 'stopId': '7820219-1', 'departure': {'delay': 0, 'time': 1601980440}}, {'stopSequence': 11, 'arrival': {'delay': 0, 'time': 1601980500}, 'stopId': '7820177-1', 'departure': {'delay': 0, 'time': 1601980500}}, {'stopSequence': 12, 'arrival': {'delay': 0, 'time': 1601980620}, 'stopId': '7820126-1', 'departure': {'delay': 0, 'time': 1601980620}}, {'stopSequence': 13, 'arrival': {'delay': 0, 'time': 1601980680}, 'stopId': '7820059-1', 'departure': {'delay': 0, 'time': 1601980680}}, {'stopSequence': 14, 'arrival': {'delay': 0, 'time': 1601980800}, 'stopId': '7820242-1', 'departure': {'delay': 0, 'time': 1601980800}}, {'stopSequence': 15, 'arrival': {'delay': 0, 'time': 1601980860}, 'stopId': '7820200-1', 'departure': {'delay': 0, 'time': 1601980860}}, {'stopSequence': 16, 'arrival': {'delay': 0, 'time': 1601980920}, 'stopId': '7820217-1', 'departure': {'delay': 0, 'time': 1601980920}}, {'stopSequence': 17, 'arrival': {'delay': 0, 'time': 1601980980}, 'stopId': '7820276-1', 'departure': {'delay': 0, 'time': 1601980980}}, {'stopSequence': 18, 'arrival': {'delay': 0, 'time': 1601981220}, 'stopId': '7820035-2', 'departure': {'delay': 0, 'time': 1601981220}}]	1	2	1011557	13:18:00	06/10/2020	IN_TRANSIT_TO	7820165-1	1	1602023361	230	58.379826	0	26.721577	15405	Live Pos
4	15405	[{'stopSequence': 1, 'arrival': {'delay': -9009, 'time': 1601970471}, 'stopId': '7820165-1', 'departure': {'delay': 0, 'time': 1601979480}}, {'stopSequence': 2, 'arrival': {'delay': 0, 'time': 1601979540}, 'stopId': '7820155-1', 'departure': {'delay': 0, 'time': 1601979540}}, {'stopSequence': 3, 'arrival': {'delay': 0, 'time': 1601979660}, 'stopId': '7820014-1', 'departure': {'delay': 0, 'time': 1601979660}}, {'stopSequence': 4, 'arrival': {'delay': 0, 'time': 1601979720}, 'stopId': '7820064-1', 'departure': {'delay': 0, 'time': 1601979720}}, {'stopSequence': 5, 'arrival': {'delay': 0, 'time': 1601979840}, 'stopId': '7820028-1', 'departure': {'delay': 0, 'time': 1601979840}}, {'stopSequence': 6, 'arrival': {'delay': 0, 'time': 1601979900}, 'stopId': '7820068-1', 'departure': {'delay': 0, 'time': 1601979900}}, {'stopSequence': 7, 'arrival': {'delay': 0, 'time': 1601980020}, 'stopId': '7820187-1', 'departure': {'delay': 0, 'time': 1601980020}}, {'stopSequence': 8, 'arrival': {'delay': 0, 'time': 1601980140}, 'stopId': '7820020-1', 'departure': {'delay': 0, 'time': 1601980140}}, {'stopSequence': 9, 'arrival': {'delay': 0, 'time': 1601980320}, 'stopId': '7820085-2', 'departure': {'delay': 0, 'time': 1601980380}}, {'stopSequence': 10, 'arrival': {'delay': 0, 'time': 1601980440}, 'stopId': '7820219-1', 'departure': {'delay': 0, 'time': 1601980440}}, {'stopSequence': 11, 'arrival': {'delay': 0, 'time': 1601980500}, 'stopId': '7820177-1', 'departure': {'delay': 0, 'time': 1601980500}}, {'stopSequence': 12, 'arrival': {'delay': 0, 'time': 1601980620}, 'stopId': '7820126-1', 'departure': {'delay': 0, 'time': 1601980620}}, {'stopSequence': 13, 'arrival': {'delay': 0, 'time': 1601980680}, 'stopId': '7820059-1', 'departure': {'delay': 0, 'time': 1601980680}}, {'stopSequence': 14, 'arrival': {'delay': 0, 'time': 1601980800}, 'stopId': '7820242-1', 'departure': {'delay': 0, 'time': 1601980800}}, {'stopSequence': 15, 'arrival': {'delay': 0, 'time': 1601980860}, 'stopId': '7820200-1', 'departure': {'delay': 0, 'time': 1601980860}}, {'stopSequence': 16, 'arrival': {'delay': 0, 'time': 1601980920}, 'stopId': '7820217-1', 'departure': {'delay': 0, 'time': 1601980920}}, {'stopSequence': 17, 'arrival': {'delay': 0, 'time': 1601980980}, 'stopId': '7820276-1', 'departure': {'delay': 0, 'time': 1601980980}}, {'stopSequence': 18, 'arrival': {'delay': 0, 'time': 1601981220}, 'stopId': '7820035-2', 'departure': {'delay': 0, 'time': 1601981220}}]	1	2	1011557	13:18:00	06/10/2020	IN_TRANSIT_TO	7820165-1	1	1602023351	230	58.379826	0	26.721577	15405	Live Pos

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

#Dir where GPS data exist
input_file = sys.argv[1]
output_file = sys.argv[2]

#Read Input .csv
print("Reading the input file")
bus_gps_df = pd.read_csv(f'{input_file}/1.csv',float_precision='round_trip')
for i in range(2,32):
    bus_gps_df_temp = pd.read_csv(f'{input_file}/{str(i)}.csv',float_precision='round_trip',low_memory=False)
    bus_gps_df = bus_gps_df.append(bus_gps_df_temp,ignore_index=True)
    

#Wrangle the data
print("Data cleaning and wrangling")
bus_gps_df = bus_gps_df\
.sort_values(by=['trip.startDate','trip.tripId','vehicle.id','timestamp'],ascending=True).reset_index(drop=True)
bus_gps_df = bus_gps_df[['trip.startDate','trip.startTime',\
                         'trip.tripId','vehicle.id','timestamp',\
                         'position.latitude','position.longitude','position.speed']]
bus_gps_df.drop_duplicates(keep='first',inplace=True,ignore_index=True)
bus_gps_df = bus_gps_df[ (bus_gps_df["position.longitude"] != 0) & (bus_gps_df["position.longitude"] !=0 ) ]\
.reset_index(drop=True)


#Function to be used in distance calculations (meters) usinf haversine formula
def haversine(lon1,lat1,lon2,lat2):
    lon1,lat1,lon2,lat2 = map(radians,[lon1,lat1,lon2,lat2])
    dlon = lon2-lon1
    dlat = lat2-lat1
    a = sin(dlat/2)**2+cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c= 2*asin(sqrt(a))
    r=6371
    return (c*r)*1000

#Function to be used in Acceleration calculation
def Average_Acceleration(speed,time):
    acceleration = []
    for i in range(1,len(speed)):
        acceleration.append((speed[i]-speed[i-1])/time[i-1] if time[i-1] > 0 else 0)
    return sum(acceleration)/len(acceleration)
    
#Segmentation process
# For each trip in the dataset , loop over the days in the track
#------------------------
print("Segmentation process started")      
days = bus_gps_df['trip.startDate'].unique()
days_group = bus_gps_df.groupby(['trip.startDate'])
#List that contain all segment
segs = []
for day in days:
    df = days_group.get_group(day)
    trips = df['trip.tripId'].unique()
    trips_group = df.groupby(['trip.tripId'])
    for trip in trips:
        df_day_trip = trips_group.get_group(trip)
        df_day_trip_list = df_day_trip.values.tolist()
        seg = []
        speed = []
        time = []   
        count = 0
        total_time = 0
        total_distance = 0
        time_start = df_day_trip_list[0][1]
        latitude_start = df_day_trip_list[0][5]         
        longitude_start   = df_day_trip_list[0][6]
        seg.append(trip)
        for index,row in enumerate(df_day_trip_list):
            #Last segment in the dataframe
            if index == len(df_day_trip_list)-1:
                time_end = row[4]
                latitude_end = row[5]    
                longitude_end = row[6] 
                count = count+1
                speed.append(row[7])
                average_speed = sum(speed)/len(speed) if len(speed) > 0 else 0
                average_acceleration = Average_Acceleration(speed,time) if len(speed) > 2 else 0
                seg.extend([day,time_start,time_end,latitude_start,longitude_start,\
                                     latitude_end,longitude_end,\
                                     count,total_time,total_distance,\
                                     average_speed,average_acceleration])
                segs.append(seg)           
            else:
                #Add GPS points to the segment
                count = count + 1
                total_time = total_time + (df_day_trip_list[index+1][4] - df_day_trip_list[index][4])
                total_distance = total_distance+ \
                haversine(df_day_trip_list[index+1][6],df_day_trip_list[index+1][5],\
                          df_day_trip_list[index][6],df_day_trip_list[index][5])           
                speed.append(row[7])   
                time.append((df_day_trip_list[index+1][4] - df_day_trip_list[index][4]))
                
# Convert the list to dataframe                 
df_segs = pd.DataFrame(segs,columns=['trip','date','time_start','time_end','latitude_start','longitude_start',\
                                     'latitude_end','longitude_end',\
                                     'point_count','total_time','total_distance',\
                                     'average_speed','average_acceleration'])
print("Segmentation process ended")

#Add transport mode as {Bus}
df_segs['Trans_mode'] = 'Bus'
df_segs = df_segs[df_segs['average_speed'] > 0]

#Save the the output to .csv file
print(f"Saving output to {output_file}")
df_segs.to_csv(path_or_buf=output_file,index=False)
print("Completed...")