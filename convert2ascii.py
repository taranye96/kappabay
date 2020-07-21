#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 13:07:38 2019

@author: amt
"""
from obspy.clients.fdsn import Client
from obspy.core import Stream, read, UTCDateTime
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from glob import glob
import numpy as np
import csv
import pandas as pd
#%%
########################################
## Paths for files:
data_dir = '/home/eking/Documents/data/corrected/'
flatfile = pd.read_csv('/home/eking/Documents/internship/data/flatfile.csv')
time = flatfile['OrgT']
#%%
def convert(s): 
  
    # initialization of string to "" 
    str1 = "" 
  
    # using join function join the list s by  
    # separating words by str1 
    return(str1.join(s)) 
      
#make a directory for each event
cortime = []
savetime = []
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
    savetime.append(convert(cortime[i]))
#%%
########################################
print('Get data')

client = Client("IRIS")

st2=Stream()
del1 = 15
del2 = 120
start = UTCDateTime(flatfile["Parr"][0]) - del1
end = start + timedelta(seconds=del2)
st2 += client.get_waveforms("BK", "*", "*", "HH*", start, end, attach_response=True)
#%%
## Get the sampling rate:
samprate = st2[0].stats['sampling_rate']

print('Remove Response')
## Make the prefilt for the instrment response - AT.SIT is @ 50Hz so 25 is nyquist
prefilt = (0.005, 0.006, ((samprate/2)-5), (samprate/2))  ## this is 20 to 25 at the end
st2.remove_response(output='VEL',pre_filt=prefilt) ## The units of data are now Velocity, m/s
#%%
print('Plot Raw data')
## Plot this unfiltered (except for instrument response) data:
plt.figure(figsize=(12,6))
plt.plot(st2[0].times(),st2[0].data)
plt.xlabel('Time from ')
plt.ylabel('Velocity (m/s)')
plt.title('Instrument Response Removed, Unfiltered, \n AT station SIT')
#plt.savefig(data_dir + 'at.sit_unfilt.png')

print('Write unfiltered response removed')
## Write the data (instr resp removed) to a SAC file, m/s
#st2.write(data_dir+'at.sit_unfilt.sac',format='SAC')
#%%
print('Filter')
## Filter the data with a highpass filter around the 13s microseism:
filtfreq = 1/13.
## Make a copy of the stream object:
st2filt = st2.copy()
## Highpass filter using 2 corners, zerophase true so it filters forwards and back
##   (so 4 corners in the end):
st2filt[0].filter('highpass',freq=filtfreq,corners=2,zerophase=True)

print('Plot filtered')
## Plot:
plt.figure(figsize=(12,6))
plt.plot(st2filt[0].times(),st2filt[0].data)
plt.xlabel('Time from Aug17_2015 UTC midnight (s)')
plt.ylabel('Velocity (m/s)')
plt.title('Instrument Response Removed, Highpass filtered 13sec, \n AT station SIT')
plt.savefig(data_dir + 'at.sit_hpfilt_13s.png')

print('Write to file')
## Write the data (instr resp removed) to a SAC file, m/s
#st2filt.write(data_dir+'at.sit_hpfilt_13s.sac',format='SAC')

print('Get stats and write to file')
## Print the station stats to a file:
stats_dict = {'network':st2filt[0].stats['network'], 'station':st2filt[0].stats['station'],
              'channel':st2filt[0].stats['channel'],'starttime':st2filt[0].stats['starttime'],
              'endtime':st2filt[0].stats['endtime'],'sampling_rate':st2filt[0].stats['sampling_rate'],
              'delta':st2filt[0].stats['delta']}

w = csv.writer(open((data_dir + 'at_sit_stats.csv'),'w'))
for key,val in stats_dict.items():
    w.writerow([key,val])
    

print('Write to txt')
## For each file in the data directory that ends in sac, read it in, extract:

all_files = glob(data_dir+'*.sac')
for file_i in range(len(all_files)):
    ## Get the whole file path:
    i_file = all_files[file_i]
    
    ## Get just the file name itself:
    i_file_name = i_file.split('/')[-1].split('.sac')[0]
    
    ## Read in the file:
    i_stream = read(i_file)
    
    ## Get channel:
    i_stream_chan = i_stream[0].stats['channel']
    
    ## Make the output file name:
    i_file_out = data_dir + i_file_name + '_' + i_stream_chan + '.csv'
    
    
    ## Get the times and data, save to a larger array:
    output_data = np.c_[i_stream[0].times(),i_stream[0].data]
    file_head = 'Time(s),Vel(m/s)'
    
    print('Writing to file ' + i_file_out)
    
    np.savetxt(i_file_out,output_data,header=file_head,delimiter=',')
    
