## Download events and stations for proposal
# VJS 5/2018
#
from obspy.clients.fdsn.client import Client
from obspy.core.utcdatetime import UTCDateTime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import os.path as path
import os

stationfile = '/home/eking/Documents/internship/data/kappa_stations2000-2018.csv'
eventfile = '/home/eking/Documents/internship/data/kappa_events2000-2018.csv'
if not path.exists(stationfile):
    os.makedirs(stationfile)
if not path.exists(eventfile):
    os.makedirs(eventfile)
Vp = 4.5
Vs = 3.5
client = Client("IRIS")
starttime = UTCDateTime("2000-01-01")
endtime = UTCDateTime("2018-05-01")
#
networks = 'BK,G,NC,NP,CE,PB,RE,SY,TO,US'
channels = 'HH*'

sta_inventory = client.get_stations(network=networks,channel=channels,starttime=starttime,endtime=endtime,minlatitude=36.3, maxlatitude=38.9, minlongitude=-123.7, maxlongitude=-121.)

#print(sta_inventory)  
#%%
## Extract stations:
sta_lon = []
sta_lat = []
sta_net = []
sta_elev = []
sta_chan = []
sta_name = []


for i_network in range(len(sta_inventory)):
    networki = sta_inventory[i_network]
    for j_station in range(len(networki)):
        stationj = networki[j_station]
        sta_lon.append(stationj.longitude)
        sta_lat.append(stationj.latitude)
        sta_elev.append(stationj.elevation)
        sta_name.append(stationj.code)
        sta_net.append(networki.code)
        sta_chan.append(channels)
      
## Print these out to a file:
station_outputdict = {'network':np.array(sta_net),'name':np.array(sta_name),'longitude':np.array(sta_lon),'latitude':np.array(sta_lat),'elevation':np.array(sta_elev),'channel':np.array(sta_chan)}
sta_df = pd.DataFrame(station_outputdict)
sta_df.to_csv(stationfile)
#%%
evClient = Client("USGS")
ev_inventory = evClient.get_events(starttime=starttime, endtime=endtime, minlatitude=36.3, maxlatitude=38.9, minlongitude=-123.7, maxlongitude=-121., minmagnitude=2.5, maxmagnitude=None)

ev_lon = []
ev_lat = []
ev_mag = []
ev_num = []
ev_depth = []
ev_time = []


for i_event in range(len(ev_inventory)):
    eventi = ev_inventory[i_event]
    ev_mag.append(eventi.magnitudes[0]['mag'])
    ev_lon.append(eventi.origins[0]['longitude'])
    ev_lat.append(eventi.origins[0]['latitude'])
    ev_depth.append(eventi.origins[0]['depth'])
    ev_num.append(str(eventi.resource_id)[57:67])
    ev_time.append(eventi.origins[0]['time'])
#%%
    
## Print these to file:
ev_outputdict = {'catalog #':np.array(ev_num),'longitude':np.array(ev_lon),'latitude':np.array(ev_lat),'magnitude':np.array(ev_mag),'depth':np.array(ev_depth), 'time':np.array(ev_time)}
ev_df = pd.DataFrame(ev_outputdict)
ev_df.to_csv(eventfile)

##map
#plt.scatter(ev_lon, ev_lat)
#plt.savefig('/Users/Elias/Documents/internship/data/figs/earthquakemap1.pdf')
## Make a figure:
#maghist,ax = plt.subplots(nrows=1,ncols=1,figsize=(7,5))
#plt.hist(ev_mag,bins=80,color='lightslategray')
#plt.yscale('log')
#plt.ylim(0.01,500)
#plt.xlim(2.4,6.3)
#plt.xlabel('Magnitude',fontsize=18)
#plt.ylabel('log10 Counts',fontsize=18)
#
#plt.setp(ax.get_xticklabels(), fontsize=14)
#plt.setp(ax.get_yticklabels(), fontsize=14)
#
#plt.text(4.5,100,'Total Number:\n %i' % len(ev_mag),fontsize=20)
#
#plt.savefig('/Users/Elias/Documents/internship/data/figs/magnitude.pdf')
#%%
timevar = ev_df['time'][31]
print(timevar)
