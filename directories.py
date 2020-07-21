# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 16:59:55 2019

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
#import waveforms as wf
import os.path as path
import os
from obspy import read
import glob


#%%

#put in your working directory here
top_dir = '/home/eking/Documents/internship/data/events/'

#can put any name directory here
new_dir = top_dir + 'unfiltered'
eventfile = '/home/eking/Documents/internship/data/kappa_events2000-2018.csv'
quakes = pd.read_csv(eventfile)
#here's a toy example, but you could put in a list of events from your file here
events = quakes['time']

def convert(s): 
  
    # initialization of string to "" 
    str1 = "" 
  
    # using join function join the list s by  
    # separating words by str1 
    return(str1.join(s)) 
      
#make a directory for each event
for i in range(len(events)):
    a = events[i]
    a = list(a)
    a[4] = "_"
    a[7] = "_"
    a[10] = "_"
    a[13] = "_"
    a[16] = "_"
    a[19] = "_"
    b = a[0:19]
    #if that event directory does not exist
    if not path.exists(new_dir + '/' + 'Event_' + convert(b)):
        os.makedirs(new_dir + '/' + 'Event_' + convert(b))
        
#%%
###########################################################################
#to make another directory full of event directories i usually use glob
event_dirs = glob.glob(top_dir + 'unfiltered/Event_*')

events = [os.path.basename(x) for x in event_dirs]#make a directory for each event
new_dir = top_dir + 'filtered'

for i in range(len(events)):
    #if that event directory does not exist
    if not path.exists(new_dir + '/' + events[i]):
        os.makedirs(new_dir + '/' + events[i])
#%%
#SNR PART
top_dir = '/home/eking/Documents/internship/data/events/'

#can put any name directory here
new_dir = top_dir + 'SNR_3/waveforms'
eventfile =   '/home/eking/Documents/internship/data/kappa_events2000-2018.csv'
quakes = pd.read_csv(eventfile)
#here's a toy example, but you could put in a list of events from your file here
events = quakes['time']

def convert(s): 
  
    # initialization of string to "" 
    str1 = "" 
  
    # using join function join the list s by  
    # separating words by str1 
    return(str1.join(s)) 
      
#make a directory for each event
for i in range(len(events)):
    a = events[i]
    a = list(a)
    a[4] = "_"
    a[7] = "_"
    a[10] = "_"
    a[13] = "_"
    a[16] = "_"
    a[19] = "_"
    b = a[0:19]
    #if that event directory does not exist
    if not path.exists(new_dir + '/' + 'Event_' + convert(b)):
        os.makedirs(new_dir + '/' + 'Event_' + convert(b))
        



       
        
        
        
    