# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 13:13:03 2019

@author: Elias
"""

#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.transforms import offset_copy
import cartopy.io.img_tiles as cimgt
from cartopy import config
import cartopy.crs as ccrs
import cartopy.feature as feature
from obspy.core.utcdatetime import UTCDateTime
import datetime
from datetime import datetime, timedelta
import pytz
import dateutil.parser

#%%
#import cmocean

stationfile = '/home/eking/Documents/internship/data/kappa_stations2000-2018.csv'
eventfile = '/home/eking/Documents/internship/data/kappa_events2000-2018.csv'

quakes = pd.read_csv(eventfile)

lat = quakes['latitude']
lon = quakes['longitude']
M = quakes['magnitude']
depth = quakes['depth']
time = quakes['time']
evnum = quakes['catalog #']

def convert(s): 
  
    # initialization of string to "" 
    str1 = "" 
  
    # using join function join the list s by  
    # separating words by str1 
    return(str1.join(s)) 
      
#make a directory for each event
cortime = []
for i in range(len(time)):
    a = time[i]
    a = list(a)
    a[4] = "_"
    a[7] = "_"
    a[10] = "_"
    a[13] = "_"
    a[16] = "_"
    a[19] = "_"
    b = a[0:22]
    cortime.append(b)
#%%
stations = pd.read_csv(stationfile)

Lat = stations['latitude']
Lon = stations['longitude']
Net = stations['network']
Name = stations['name']
Elev = stations['elevation']
Chan = stations['channel']
#%%
def compute_rhyp(stlon,stlat,stelv,hypolon,hypolat,hypodepth):
    '''
    Compute the hypocentral distance for a given station lon, lat and event hypo
    Input:
        stlon:          Float with the station/site longitude
        stlat:          Float with the station/site latitude
        stelv:          Float with the station/site elevation 
        hypolon:        Float with the hypocentral longitude
        hypolat:        Float with the hypocentral latitude
        hypodepth:      Float with the hypocentral depth 
    Output:
        rhyp:           Float with the hypocentral distance, in km
    '''
    
    import numpy as np
    from pyproj import Geod
    
    
    ## Make the projection:
    p = Geod(ellps='WGS84')
    
    ## Apply the projection to get azimuth, backazimuth, distance (in meters): 
    az,backaz,horizontal_distance = p.inv(stlon,stlat,hypolon,hypolat)

    ## Put them into kilometers:
    horizontal_distance = horizontal_distance/1000.
    stelv = float(stelv)
    stelv = stelv/1000
    ## Hypo deptha lready in km, but it's positive down. ST elevation is positive
    ##    up, so make hypo negative down (so the subtraction works out):
    hypodepth = float(hypodepth)
    hypodepth = hypodepth/1000 * -1
    
    ## Get the distance between them:
    rhyp = np.sqrt(horizontal_distance**2 + (stelv - hypodepth)**2)
    
    return rhyp
#%%
phatlist = []
for i in range(len(Lat)):
    for j in range(len(lat)):
        phatlist.append(Net[i])
        phatlist.append(Name[i])
        phatlist.append(Lat[i])
        phatlist.append(Lon[i])
        phatlist.append(Elev[i])
        phatlist.append(evnum[j])
        phatlist.append(M[j])
        phatlist.append(lat[j])
        phatlist.append(lon[j])
        phatlist.append(depth[j])
        phatlist.append(convert(cortime[j]))
    
print(len(phatlist))
#%%
phatarray = np.array(phatlist)
flatfilearray = np.vstack(np.array_split(phatarray, 121664))
ffd = pd.DataFrame(flatfilearray, columns=['Network', 'Name', 'Slat', 'Slon', 'Selv', 'Quake#', 'Mag', 'Qlat', 'Qlon', 'Qdep', 'OrgT'])
#%%
rhyplist = []

for i in range(121664):
    dist = compute_rhyp(ffd['Slon'][i], ffd['Slat'][i], ffd['Selv'][i], ffd['Qlon'][i], ffd['Qlat'][i], ffd['Qdep'][i])
    rhyplist.append(dist)

ffd['rhyp']= rhyplist
#%%
for i in range(121664):
    
    #ffd['OrgT'][i] = UTCDateTime(ffd['OrgT'][i])
    ffd['OrgT'][i] = datetime.strptime(ffd['OrgT'][i],"%Y_%m_%d_%H_%M_%S_%f")  #just in case
    if i % 100 == 0:
        print(i)

#%%
def trav_time(v, dist):
    time = dist/v
    return time
Vs = 3.5
Vp = 4.5

Ptrav = []
Strav = []

for i in range(121664):
    Pt = trav_time(Vp, ffd['rhyp'][i])
    Vt = trav_time(Vs, ffd['rhyp'][i])
    Ptrav.append(Pt)
    Strav.append(Vt)



#%%
Parr = []
Sarr = []
for i in range(121664):
    t = ffd['OrgT'][i]
    p = t + timedelta(seconds=Ptrav[i])
    s = t + timedelta(seconds=Strav[i])
    Parr.append(p)
    Sarr.append(s)

ffd['Parr']= Parr
ffd['Sarr']= Sarr
#%%
#for i in range(121664):
    

#%%
#Vs30 = pd.read_csv('/Users/Elias/Documents/internship/data/vs30data (1).csv')
#%%
ffd.to_csv('/home/eking/Documents/internship/data/flatfile.csv')
#%%

                                           
                                           
                                           