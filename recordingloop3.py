#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 15:57:35 2019

@author: eking
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
data_dir1 = '/home/eking/Documents/internship/data/events/unfiltered/'
data_dir2 = '/home/eking/Documents/internship/data/events/filtered/'
allmetadata = pd.read_csv('/home/eking/Documents/internship/data/flatfile3.csv')
count = 0
nummissed = 0
client = Client("NCEDC")
ffl = []
td = 30

#%%
#i_line = 2097
count = 0 
nummissed = 0
ffl = []
for i_line in range(len(allmetadata)):
    
# first set up all the parameters in the loop
    
    i_date,i_time = allmetadata.loc[i_line]['OrgT'].split('T')
    i_year,i_month,i_day = i_date.split('-')
    i_hr,i_min,i_sec = i_time.split(':')
    i_sec,_ = i_sec.split('.')
    i_parr,_ = allmetadata.loc[i_line]['Parr'].split('.')
    i_sarr,_ =  allmetadata.loc[i_line]['Sarr'].split('.')
    i_network = allmetadata.loc[i_line]['Network']
    i_Parr = datetime.strptime(i_parr,"%Y-%m-%dT%H:%M:%S")
    i_Sarr =datetime.strptime(i_sarr,"%Y-%m-%dT%H:%M:%S")
    i_station = allmetadata.loc[i_line]['Name']
    sp = i_Sarr - i_Parr
    i_start = i_Parr - timedelta(seconds=td)
    i_end = i_Sarr + timedelta(seconds=90)
    i_network = str(i_network)
    i_station = str(i_station)
########################################################################################################################################3
# chlear stream objects 
    raw_stn = Stream()
    raw_ste = Stream()
#data download and plot for Nort
    try:
        raw_stn += client.get_waveforms(i_network, i_station, "*", 'HHN', UTCDateTime(i_start), UTCDateTime(i_end), attach_response=True)
    except:
        nummissed += .5
        print('oops')
        continue
    i_stn = raw_stn.copy()
    j_stn = raw_stn.copy()
    
    
    samprate = i_stn[0].stats['sampling_rate']
    ## Make the prefilt for the instrment response - AT.SIT is @ 50Hz so 25 is nyquist
    prefilt1 = (0.005, 0.006, ((samprate/2)-5), (samprate/2))  ## this is 20 to 25 at the end
    i_stn[0].remove_response(output='VEL',pre_filt=prefilt1) ## The units of data are now Velocity, m/s
    prefilt2 = (0.063, 0.28,((samprate/2)-5), (samprate/2)) ## 0.063 just above 16 sec microsiesem, .28 just above 4 sec
    j_stn[0].remove_response(output='VEL',pre_filt=prefilt2) ## The units of data are now Velocity, m/s
    
    ## make and save plot
    plt.figure(figsize=(12,6))
    plt.plot(i_stn[0].times(),i_stn[0].data,'g')
    plt.axvline(x=td)
    plt.axvline(x=td+sp.seconds)
    plt.xlabel('Time from ')
    plt.ylabel('Velocity (m/s)')
    plt.title('Instrument Response Removed, Unfiltered, \n' + i_network + i_station )
    plt.savefig(data_dir1 + 'Event'+'_'+i_year+'_'+i_month+'_'+i_day+'_'+i_hr+'_'+i_min+'_'+i_sec+'/'+i_network+'_'+i_station+'_'+'HHN'+'_'+i_year+'_'+i_month+'_'+i_day+'_'+i_hr+'_'+i_min+'.png')
    plt.close('all')
    i_stn[0].write(data_dir1+'Event'+'_'+i_year+'_'+i_month+'_'+i_day+'_'+i_hr+'_'+i_min+'_'+i_sec+'/'+i_network+'_'+i_station+'_'+'HHN'+'_'+i_year+'_'+i_month+'_'+i_day+'_'+i_hr+'_'+i_min+'.sac',format='SAC')
    #ffl.append(i_stn[0])
    plt.figure(figsize=(12,6))
    plt.plot(j_stn[0].times(),j_stn[0].data,'g')
    plt.axvline(x=td)
    plt.axvline(x=td+sp.seconds)
    plt.xlabel('Time from ')
    plt.ylabel('Velocity (m/s)')
    plt.title('Instrument Response Removed, filtered, \n' + i_network +i_station )
    plt.savefig(data_dir2 + 'Event'+'_'+i_year+'_'+i_month+'_'+i_day+'_'+i_hr+'_'+i_min+'_'+i_sec+'/'+i_network+'_'+i_station+'_'+'HHN'+'_'+i_year+'_'+i_month+'_'+i_day+'_'+i_hr+'_'+i_min+'.png')
    plt.close('all')
    j_stn[0].write(data_dir2+'Event'+'_'+i_year+'_'+i_month+'_'+i_day+'_'+i_hr+'_'+i_min+'_'+i_sec+'/'+i_network+'_'+i_station+'_'+'HHN'+'_'+i_year+'_'+i_month+'_'+i_day+'_'+i_hr+'_'+i_min+'.sac',format='SAC')
    
    
######################################################################################################################################################################################
#datqa dow2nload and plot for east
    try:
        raw_ste += client.get_waveforms(i_network, i_station, "*", 'HHE', UTCDateTime(i_start), UTCDateTime(i_end), attach_response=True)
    except:
        nummissed += .5
        print('oops')
        continue
    i_ste = raw_ste.copy()
    j_ste = raw_ste.copy()
    
    
    samprate = i_ste[0].stats['sampling_rate']
    ## Make the prefilt for the instrment response - AT.SIT is @ 50Hz so 25 is nyquist
    prefilt1 = (0.005, 0.006, ((samprate/2)-5), (samprate/2))  ## this is 20 to 25 at the end
    i_ste[0].remove_response(output='VEL',pre_filt=prefilt1) ## The units of data are now Velocity, m/s
    prefilt2 = (0.063,0.28,((samprate/2)-5), (samprate/2)) ## 0.063 just above 16 sec microsiesem, .28 just above 4 sec
    j_ste[0].remove_response(output='VEL',pre_filt=prefilt2) ## The units of data are now Velocity, m/s
    
    ## make and save plot
    plt.figure(figsize=(12,6))
    plt.plot(i_ste[0].times(),i_ste[0].data,'g')
    plt.axvline(x=td)
    plt.axvline(x=td+sp.seconds)
    plt.xlabel('Time from ')
    plt.ylabel('Velocity (m/s)')
    plt.title('Instrument Response Removed, Unfiltered, \n' + i_network + i_station )
    plt.savefig(data_dir1 + 'Event'+'_'+i_year+'_'+i_month+'_'+i_day+'_'+i_hr+'_'+i_min+'_'+i_sec+'/'+i_network+'_'+i_station+'_'+'HHE'+'_'+i_year+'_'+i_month+'_'+i_day+'_'+i_hr+'_'+i_min+'.png')
    plt.close('all')
    i_ste[0].write(data_dir1+'Event'+'_'+i_year+'_'+i_month+'_'+i_day+'_'+i_hr+'_'+i_min+'_'+i_sec+'/'+i_network+'_'+i_station+'_'+'HHE'+'_'+i_year+'_'+i_month+'_'+i_day+'_'+i_hr+'_'+i_min+'.sac',format='SAC')
    #ffl.append(i_stn[0])
    plt.figure(figsize=(12,6))
    plt.plot(j_ste[0].times(),j_ste[0].data,'g')
    plt.axvline(x=td)
    plt.axvline(x=td+sp.seconds)
    plt.xlabel('Time from ')
    plt.ylabel('Velocity (m/s)')
    plt.title('Instrument Response Removed, filtered, \n' + i_network +i_station )
    plt.savefig(data_dir2 + 'Event'+'_'+i_year+'_'+i_month+'_'+i_day+'_'+i_hr+'_'+i_min+'_'+i_sec+'/'+i_network+'_'+i_station+'_'+'HHE'+'_'+i_year+'_'+i_month+'_'+i_day+'_'+i_hr+'_'+i_min+'.png')
    plt.close('all')
    j_ste[0].write(data_dir2+'Event'+'_'+i_year+'_'+i_month+'_'+i_day+'_'+i_hr+'_'+i_min+'_'+i_sec+'/'+i_network+'_'+i_station+'_'+'HHE'+'_'+i_year+'_'+i_month+'_'+i_day+'_'+i_hr+'_'+i_min+'.sac',format='SAC')
    
    
#   
####################################################################################################################################################################
    
    
   
    
    
    print(i_line)
#    count +=1
#    if count == 10:
#        break


print(nummissed)
    
