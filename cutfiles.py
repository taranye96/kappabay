#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 11:56:54 2019

@author: eking
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.transforms import offset_copy
import cartopy.io.img_tiles as cimgt
from cartopy import config
import cartopy.crs as ccrs
import cartopy.feature as feature
from obspy.core.utcdatetime import UTCDateTime
import obspy
import datetime
from datetime import datetime, timedelta
import pytz
import dateutil.parser
import seaborn as sns; sns.set(style="white", color_codes=True)
import glob
import os
import os.path as path
import time
import kappa_utils as ku


#%%

working_dir = '/home/eking/Documents/internship/data/Kappa/SNR_5'

#path to corrected seismograms
event_dirs = glob.glob(working_dir + '/waveforms/Event_*')
cutstart = 30
cutlength = 60
outpath = working_dir + '/cutWF_' + np.str(cutlength)

events = []
for i in range(len(event_dirs)):
    events.append(path.basename(event_dirs[i]))
for i in range(len(events)):
    if not path.exists(outpath + '/' + events[i]):
        os.makedirs(outpath + '/'  + events[i])

for i in range(len(event_dirs)):
    t1 = time.time()
    event = events[i][6:]
    print(i)
    print('cutting: '+ event)
    recordpaths = glob.glob(working_dir + '/waveforms/Event_' + event +'/*_*_HHN_' + event + '.sac')#full path for only specified channel
    stns = [(x.split('/')[-1]).split('_')[1] for x in recordpaths]
    print(stns)
    for j in range(len(stns)):
        recordpath_E = glob.glob(working_dir + '/waveforms/Event_' + event +'/*_' + stns[j] + '_HHE_' + event + '.sac')
        recordpath_N = glob.glob(working_dir + '/waveforms/Event_' + event +'/*_' + stns[j] + '_HHN_' + event + '.sac')
        if(len(recordpath_E) == 1 and len(recordpath_N) == 1):
            #North component
            
            outpath_E = recordpath_E[0].replace('waveforms','cutWF_' + np.str(cutlength))
            
            ku.cut_swave(recordpath_E[0], outpath_E, cutstart, cutlength)
            
            
            outpath_N = recordpath_N[0].replace('waveforms','cutWF_' + np.str(cutlength))
            
            ku.cut_swave(recordpath_N[0], outpath_N, cutstart, cutlength)
            
    