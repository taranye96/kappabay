#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 13:16:55 2019

@author: eking
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
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
import seaborn as sns; sns.set(style="darkgrid", color_codes=True)
import kappa_utils
import pickle
#%%
    

working_dir =  '/home/eking/Documents/internship/data/Kappa/'

event_dir = working_dir + 'SNR_5/75_bins/15/'


#stations 
BBGB = []
BDM = []
BKS =[]
BL67 = []
BRIB =[]
BRK = []
CVS = []
DIX = []
ELK = []
EMR = []
FARB = []
GDXB = []
HAST = []
JRSC = []
KIR = []
MCCM = []
MHC = []
MHDL = []
MNRC = []
MOD = []
MTOS = []
OXMT = []
PACP = []
PLA = []
POTR = []
SAO = []
SCZ = []
SIA = []
VAK = []
WENL = []
kappaFL = []
station_list = [BBGB,BDM,BKS,BL67,BRIB,BRK,CVS,DIX,ELK,EMR,FARB,GDXB,HAST,JRSC,KIR,MCCM,MHC,MHDL,MNRC,MOD,MTOS,OXMT,PACP,PLA,POTR,SAO,SCZ,SIA,VAK,WENL]
station_str_list = ['BBGB','BDM','BKS','BL67','BRIB','BRK','CVS','DIX','ELK','EMR','FARB','GDXB','HAST','JRSC','KIR','MCCM','MHC','MHDL','MNRC','MOD','MTOS','OXMT','PACP','PLA','POTR','SAO','SCZ','SIA','VAK','WENL']

#fig_dirs = run_dirs + '/figs_*'
#fill in constraint event
con_dir = 'Andrews_inversion_constrained_*'
print(event_dir)
#%%

kappafile_path = event_dir + 'kappa_site.out'
#kappaFL.append(kappafile)
con_stn = glob.glob(event_dir + '/'+ con_dir + '/[!2]*.out')


print(con_stn)
#%%
for i in range(len(con_stn)):#for each station
    #make each row into an array
    #stn = np.genfromtxt(con_stn[i])
    stnid = path.basename(con_stn[i]).split('.')[0]
   
    if stnid == 'BBGB':
        BBGB.append(con_stn[i])
    if stnid == 'BDM':
        BDM.append(con_stn[i])
    if stnid == 'BKS':
        BKS.append(con_stn[i])
    if stnid == 'BL67':
        BL67.append(con_stn[i])
    if stnid == 'BRIB':
        BRIB.append(con_stn[i])
    if stnid == 'BRK':
        BRK.append(con_stn[i])
    if stnid == 'CVS':
        CVS.append(con_stn[i])
    if stnid == 'DIX':
        DIX.append(con_stn[i])
    if stnid == 'ELK':
        ELK.append(con_stn[i])
    if stnid == 'EMR':
        EMR.append(con_stn[i])
    if stnid == 'FARB':
        FARB.append(con_stn[i])
    if stnid == 'GDXB':
        GDXB.append(con_stn[i])
    if stnid == 'HAST':
        HAST.append(con_stn[i])
    if stnid == 'JRSC':
        JRSC.append(con_stn[i])
    if stnid == 'KIR':
        KIR.append(con_stn[i])
    if stnid == 'MCCM':
        MCCM.append(con_stn[i])
    if stnid == 'MHC':
        MHC.append(con_stn[i])
    if stnid == 'MHDL':
        MHDL.append(con_stn[i])
    if stnid == 'MNRC':
        MNRC.append(con_stn[i])
    if stnid == 'MOD':
        MOD.append(con_stn[i])
    if stnid == 'MTOS':
        MTOS.append(con_stn[i])
    if stnid == 'OXMT':
        OXMT.append(con_stn[i])
    if stnid == 'PACP':
        PACP.append(con_stn[i])
    if stnid == 'PLA':
        PLA.append(con_stn[i])
    if stnid == 'POTR':
        POTR.append(con_stn[i])
    if stnid == 'SAO':
        SAO.append(con_stn[i])
    if stnid == 'SCZ':
        SCZ.append(con_stn[i])
    if stnid == 'SIA':
        SIA.append(con_stn[i])
    if stnid == 'VAK':
        VAK.append(con_stn[i])
    if stnid == 'WENL':
        WENL.append(con_stn[i])
#%%
from math import log10, floor
def round_sig(x, sig=3):
    return round(x, sig-int(floor(log10(abs(x))))-1)
#%%
cmap = plt.cm.hsv  # define the colormap
# extract all colors from the .jet map
cmaplist = [cmap(i) for i in range(cmap.N)]
# force the first color entry to be grey


# create the new map
cmap = mpl.colors.LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist, 30)

cmaplist = [cmap(i) for i in range(cmap.N)]

# define the bins and normalize
bounds = np.linspace(0, 30, 31)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)


#%%
fig = plt.figure(figsize = (12,10))
plt.axes().set_xscale("log")
plt.axes().set_yscale("log")
plt.ylabel('Velocity amplitude (m)', fontsize = 16)
plt.xlabel('Frequency (Hz)', fontsize = 16)
plt.xlim(0.5,50)
plt.ylim(10e-6,3)
plt.minorticks_on
plt.grid(True, which='both')
plt.tick_params(axis='both', which='major', labelsize=15)
plt.tick_params(axis='both', which='both', length = 5, width = 1)
plt.title('All Station Spectra SNR 5 75 bins 15 sec cut', fontsize = 16)
kappafile = pd.read_csv(kappafile_path)

for i in range(len(station_list)):
       
    
    stn = np.genfromtxt(con_stn[i])
    stnid = station_str_list[i]
    ind = int(np.array(np.where(kappafile['Name'] == stnid)))
    kappa = round_sig(kappafile[' tstar(s) '][ind])
    std = round_sig(kappafile[' tstar_std '][ind])
    
    print(stn)
    
    freq = stn.T[0]
    amp = stn.T[1]
    std = stn.T[2]
    plt_ind = np.argmin(np.abs(freq - 35))
    
    plt.errorbar(freq[0:plt_ind] , amp[0:plt_ind], yerr=std[0:plt_ind],  c= cmaplist[i],label= stnid)#+ 'plus or minus'+ std)
    #plt.text(20,10**-1.1,kappa)
       
    #plt.loglog(freq, Brune_list[ind], color = 'blue', label = 'Brune spectra')
    plt.legend(loc = 'lower left', fontsize = 16)
       


   # plt.text(0.7, .1, 'Median log(diff) 1-32.7 Hz (demeaned): ' + str(round(sum_list[ind],3)), fontsize = 16)
plt.savefig('/home/eking/Documents/internship/data/Kappa/station_spectra_all.png')
plt.show()
#        