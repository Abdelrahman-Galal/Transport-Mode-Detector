'''
This script is used for data cleaning and segmentation of MobilityLog GPS App (Tartu Mobility Lab)
Input: .csv file contains GPS data in the below format
====
id	counter	time_system	time_gps	time_system_ts	time_gps_ts	point	accuracy	altitude	bearing	speed	date	user_id_new
753434928	5391998	1.56837E+12	1.56837E+12	2019-09-13 13:56:27+03	2019-09-13 13:56:28+03	0101000020E61000000000D0C6C1E739400000A49630594D40	7	82	340.2	25.08	13/09/2019	7597
752350957	5353118	1.56784E+12	1.56784E+12	2019-09-07 11:23:45+03	2019-09-07 11:23:46+03	0101000020E61000000000B016A09E38400000A6C933A84D40	19	53	81.1	29.62	07/09/2019	7597
752062289	5331534	1.56768E+12	1.56768E+12	2019-09-05 13:05:16+03	2019-09-05 13:05:18+03	0101000020E6100000000000429FB83A4000005E0CA2304D40	8	46	130.1	1.06	05/09/2019	7597
753434915	5391972	1.56837E+12	1.56837E+12	2019-09-13 13:56:14+03	2019-09-13 13:56:15+03	0101000020E61000000000385B3DE839400000782CD6584D40	8	83	340	24.86	13/09/2019	7597
755167669	5428486	1.56925E+12	1.56925E+12	2019-09-23 17:48:29+03	2019-09-23 17:48:30+03	0101000020E61000000000C8A11BBA3A400000EEB12F314D40	59	63	259.8	2.71	23/09/2019	7597
752062283	5331522	1.56768E+12	1.56768E+12	2019-09-05 13:05:10+03	2019-09-05 13:05:12+03	0101000020E61000000000483E97B83A4000009ECBA2304D40	11	45	129.8	1.22	05/09/2019	7597
755167677	5428502	1.56925E+12	1.56925E+12	2019-09-23 17:48:37+03	2019-09-23 17:48:38+03	0101000020E6100000000018B224BA3A40000060AB2D314D40	64	78	287.5	1.4	23/09/2019	7597
====
Output: .csv file contain cleaned and segmented data
'''

#Import required libraries
from datetime import datetime
from datetime import timedelta
from math import radians,cos,sin,asin,sqrt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely import geometry, wkb
import sys
import os

input_file = sys.argv[1]
output_file = sys.argv[2]

#Read Input .csv
print("Reading the input file")
mobility_gps_df = pd.read_csv("data_gps.csv",float_precision='round_trip')
mobility_gps_df['latitude'] = mobility_gps_df.point.apply(lambda z : wkb.loads(z,hex=True).y)
mobility_gps_df['longitude'] = mobility_gps_df.point.apply(lambda z : wkb.loads(z,hex=True).x)
#Tartu borders
mobility_gps_df = mobility_gps_df[ (mobility_gps_df.latitude >= 58.340155) & (mobility_gps_df.latitude <= 58.407981) & \
              (mobility_gps_df.longitude >= 26.671466) & (mobility_gps_df.longitude <= 26.798375 )  ]  

#Format the time
FMT = '%H:%M:%S'
mobility_gps_df['coord_date'] = mobility_gps_df['time_gps_ts'].str[0:10]
mobility_gps_df['coord_time'] = mobility_gps_df['time_gps_ts'].str[11:19]
mobility_gps_df['coord_time'] = pd.to_datetime(mobility_gps_df['coord_time'],format=FMT)
mobility_gps_df = mobility_gps_df.sort_values(by=['user_id_new','coord_date','coord_time'],ascending=True)\
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
mobility_gps_df['date_start'] = mobility_gps_df['coord_date']
mobility_gps_df['date_end'] = mobility_gps_df['coord_date'].iloc[1:].reset_index(drop=True)
mobility_gps_df['time_start'] = mobility_gps_df['coord_time']
mobility_gps_df['time_end'] = mobility_gps_df['coord_time'].iloc[1:].reset_index(drop=True)
mobility_gps_df['latitude_start'] = mobility_gps_df['latitude']
mobility_gps_df['longitude_start'] = mobility_gps_df['longitude']
mobility_gps_df['latitude_end'] = mobility_gps_df['latitude_start'].iloc[1:].reset_index(drop=True)
mobility_gps_df['longitude_end'] = mobility_gps_df['longitude_start'].iloc[1:].reset_index(drop=True)
mobility_gps_df['user'] = mobility_gps_df['user_id_new']
mobility_gps_df['user_check'] = mobility_gps_df['user'].iloc[1:].reset_index(drop=True)
mobility_gps_df = mobility_gps_df.\
        drop(['id','counter','accuracy','altitude','bearing',\
              'user_id_new','date','point','time_system',\
              'time_system_ts','time_gps_ts','time_gps'
              ,'coord_date','coord_time','latitude','longitude','speed'],axis='columns')
mobility_gps_df = mobility_gps_df.drop([len(mobility_gps_df) - 1], axis='index')
column_names = []


#Keep only rows that has the same start and end data and belongs to the same user              
mobility_gps_df = mobility_gps_df[(mobility_gps_df.date_start == mobility_gps_df.date_end) \
                          & (mobility_gps_df.user == mobility_gps_df.user_check)].reset_index(drop=True)



#Calculate time delta in seconds between each congestive points
print("Time delta calculation")
mobility_gps_df['time_diff'] = mobility_gps_df['time_end'] - mobility_gps_df['time_start']
time_diff = mobility_gps_df['time_diff'].to_list()
time_diff_seconds = [timedelta.total_seconds(diff) for diff in time_diff]
mobility_gps_df['time_diff_seconds'] = time_diff_seconds


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
longitude_start = mobility_gps_df['longitude_start'].to_list()
latitude_start = mobility_gps_df['latitude_start'].to_list()
longitude_end = mobility_gps_df['longitude_end'].to_list()
latitude_end = mobility_gps_df['latitude_end'].to_list()
mobility_gps_df['distance' ]=[haversine(lon1,lat1,lon2,lat2) \
           for lon1,lat1,lon2,lat2 in zip(longitude_start,latitude_start,longitude_end,latitude_end)]


#Drop points with zero time delta
mobility_gps_df   = mobility_gps_df[ mobility_gps_df['time_diff_seconds'] != 0 ].reset_index(drop=True)


#Calculate speed (meters/second)
mobility_gps_df = mobility_gps_df [ mobility_gps_df['time_diff_seconds'] < 60 ].reset_index(drop=True)
mobility_gps_df = mobility_gps_df[mobility_gps_df['distance'] <= 40* mobility_gps_df['time_diff_seconds']]\
.reset_index(drop=True)
mobility_gps_df['speed'] = mobility_gps_df['distance']/mobility_gps_df['time_diff_seconds']

#Classification Walking and non-Walking points (1 meter/second) thershould
print("Classification of Walking and non-Walking GPS points")
speed = mobility_gps_df['speed'].to_list()
Trans_mode = [ "Walking" if row < 1 else "Non-Walking" for row in speed]
mobility_gps_df['Trans_mode'] = Trans_mode
mobility_gps_df['Trans_mode_Check'] = mobility_gps_df['Trans_mode'].iloc[1:].reset_index(drop=True)
mobility_gps_df = mobility_gps_df.drop([len(mobility_gps_df) - 1], axis='index')

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
users = mobility_gps_df['user'].unique()
user_group = mobility_gps_df.groupby(['user'])
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
        trans_mode = ""
        points = []
        seg.append(user)
        for index,row in enumerate(df_user_day_list):
            #End of segment if the transmode changed between walking to non-walking or vice versa   
            if row[14] != row[15]:
                time_end = row[2]
                latitude_end = row[4]    
                longitude_end = row[5]
                average_speed = sum(speed)/len(speed) if len(speed) > 0 else 0
                average_acceleration = Average_Acceleration(speed,time) if len(speed) > 2 else 0
                count = count + 1
                points.append((latitude_end,longitude_end))
                seg.extend([day,time_start,time_end,latitude_start,longitude_start,\
                            latitude_end,longitude_end,\
                            count,total_time,total_distance,\
                            average_speed,average_acceleration,points,row[14]])
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
                trans_mode = ""
                points = []
                try:
                    time_start = df_user_day_list[index+1][2]
                    latitude_start = df_user_day_list[index+1][4]         
                    longitude_start   = df_user_day_list[index+1][5]
                except:
                    pass
            #last row 
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
                points.append((latitude_end,longitude_end))
                seg.extend([day,time_start,time_end,latitude_start,longitude_start,\
                                     latitude_end,longitude_end,\
                                     count,total_time,total_distance,\
                                     average_speed,average_acceleration,points,row[14]])
                segs.append(seg)
            #Add GPS points to the segment                
            else:
                count = count + 1
                total_time = total_time + row[11]
                total_distance = total_distance+row[12]            
                speed.append(row[13])   
                time.append(row[11])
                trans_mode = row[14]
                points.append((row[4],row[5]))

# Convert the list to dataframe                 
df_segs = pd.DataFrame(segs,columns=['user','date','time_start','time_end','latitude_start','longitude_start',\
                                     'latitude_end','longitude_end',\
                                     'point_count','total_time','total_distance',\
                                     'average_speed','average_acceleration','points','Trans_mode'])
print("Segmentation process ended")

#Drop short Walking segments (possible traffic light or bus stop )
df_segs = df_segs[ ~ ((df_segs['point_count'] <= 5 ) | (df_segs['total_time'] <= 90 ) \
                  & (df_segs['Trans_mode'] == 'Walking')) ].reset_index(drop=True)
                  
print(df_segs.Trans_mode.value_counts())
#Merge non-walking segments if same user  , same date and small time diff
print("Merge non-Walkinging segments process")
nonwalking_segs = df_segs[df_segs['Trans_mode'] == 'Non-Walking'].values.tolist()
new_segs = []
new_segs.append(nonwalking_segs[0][:-1])
for i in range(1,len(nonwalking_segs)):
    #merge if same user  , same date and small time diff
    if ( nonwalking_segs[i][0] == new_segs[-1][0] \
       and nonwalking_segs[i][1] == new_segs[-1][1]\
       and (nonwalking_segs[i][2] - new_segs[-1][3]).seconds < 60):
        tmp = []
        user = nonwalking_segs[i][0]
        date = nonwalking_segs[i][1]
        time_start = new_segs[-1][2]
        time_end =nonwalking_segs[i][3]
        latitude_start = new_segs[-1][4]
        longitude_start = new_segs[-1][5]
        latitude_end = nonwalking_segs[i][6]
        longitude_end = nonwalking_segs[i][7]
        point_count = nonwalking_segs[i][8] + new_segs[-1][8]
        total_time = nonwalking_segs[i][9] + new_segs[-1][9]
        total_distance = nonwalking_segs[i][10] + new_segs[-1][10]
        average_speed = ( nonwalking_segs[i][8] * nonwalking_segs[i][11] \
                         + new_segs[-1][8] * new_segs[-1][11] ) /  point_count
        average_acceleration = (  nonwalking_segs[i][8] * nonwalking_segs[i][12] \
                         + new_segs[-1][8] * new_segs[-1][12] ) /  point_count
        points = new_segs[-1][13] + nonwalking_segs[i][13]
        tmp = [user,date,time_start,time_end,latitude_start,longitude_start\
        ,latitude_end,longitude_end,point_count,total_time,total_distance,average_speed,average_acceleration,points]
        new_segs.pop()
        new_segs.append(tmp)
         
    else:
        new_segs.append(nonwalking_segs[i][0:-1])       
 
# Convert the list of non-walking segments to a dataframe 
df_segs_unlabeled = pd.DataFrame(new_segs,columns=['user','date','time_start','time_end','latitude_start',\
                                     'longitude_start',\
                                     'latitude_end','longitude_end',\
                                     'point_count','total_time','total_distance',\
                                     'average_speed','average_acceleration','points'])

#Save the the output to .csv file
print(f"Saving output to {output_file}")                                     
df_segs_unlabeled.to_csv(path_or_buf=output_file,index=False)
print("Completed...") 