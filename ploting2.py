#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 12:40:56 2019

@author: eking
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
import seaborn as sns; sns.set(style="darkgrid", color_codes=True)
import kappa_utils
import pickle

#%%
#kappa by station

working_dir =  '/home/eking/Documents/internship/data/Kappa/'

full_file_paths = glob.glob(working_dir + 'SNR_*/*_bins/*/full_file.csv')

#print(full_file_paths)

data = pd.read_csv(full_file_paths[5])
##x = data['Name']
#run5y = data[' tstar(s) ']
std5 = data[' tstar_std ']
#%%
fig = plt.figure(figsize = (15,10))
plt.ylabel('Kappa', fontsize = 16)
plt.xlabel('Station', fontsize = 16)
plt.minorticks_on
plt.grid(True, which='both')
plt.tick_params(axis='both', which='major', labelsize=12, rotation=90)
plt.tick_params(axis='both', which='both', length = 5, width = 1)
plt.title('Kappa vs station' , fontsize = 16)

plt.errorbar(x,run0y,yerr=std0, c='b',fmt=' ',capsize=8, elinewidth=1, marker='o',label='75 bins, 60 sec, snr 3')
plt.errorbar(x,run1y,yerr=std1, c='r',fmt=' ',capsize=8, elinewidth=1,  marker='^',label='75 bins, 15 sec, snr 3')
plt.errorbar(x,run2y,yerr=std2, c='g',fmt=' ',capsize=8, elinewidth=1,  marker='s',label='150 bins, 15 sec, snr 3')
plt.errorbar(x,run3y,yerr=std3, c='y',fmt=' ',capsize=8, elinewidth=1,  marker='*',label='75 bins, 60 sec, snr 5')
plt.errorbar(x,run4y,yerr=std4, c='c',fmt=' ',capsize=8, elinewidth=1,  marker='h',label='75 bins, 15 sec, snr 5')
plt.errorbar(x,run5y,yerr=std5, c='m',fmt=' ',capsize=8, elinewidth=1,  marker='p',label='150 bins, 15 sec, snr 5')
plt.legend(fontsize = 12)

#%%
#vs30 plot 
from scipy.stats.stats import pearsonr
from statsmodels.stats.power import tt_ind_solve_power

bins= 75
cut = 15
snr = 5
vs30_path = '/home/eking/Documents/internship/data/Kappa/vs30.csv'
full_file_path =  '/home/eking/Documents/internship/data/Kappa/SNR_'+np.str(snr)+'/'+np.str(bins)+'_bins/'+np.str(cut)+'/full_file.csv'
savepath = '/home/eking/Documents/internship/data/Kappa/K_vs_Vs30/SNR_'+np.str(snr)+'_'+np.str(bins)+'_bins_'+np.str(cut)+'log.png'
def plot_KvsVs30(bins, cut, snr, vs30_path, full_file_path,savepath):  
    fullfile = pd.read_csv(full_file_path)
    
    kappa = fullfile[' tstar(s) '].values
    vs30 = pd.read_csv(vs30_path).values
    name = fullfile['Name'].values
    vs30_1d = np.reshape(vs30,-1)
    lnvs30 = np.log(vs30_1d)
    print(lnvs30)
    Pearson_R,P_val = pearsonr(kappa,vs30_1d)
    Power = tt_ind_solve_power(effect_size=Pearson_R,nobs1=30, alpha=0.05)
    Pearson_R = float(Pearson_R)
    P_val = float(P_val)
    Power = float(Power)
      #  text = 'Pearson R = ' + r + '   P-val = ' + p_val +'   Power = ' + power + '       .'
    textstr = '\n'.join((
    r'$Pearson R:%.4f$' % (Pearson_R, ),
    r'$P val:%.4f$' % (P_val, ),
    r'$Power:%.4f$' % (Power, )))
    ##########################
    fig = plt.figure(figsize = (10,10))
    #    plt.axes().set_xscale("log")
    #    plt.axes().set_yscale("log")
    plt.ylabel('ln(Vs30) (m/s)', fontsize = 16)
    plt.xlabel('Kappa (s)', fontsize = 16)
    #plt.xlim(0.5,50)
    #plt.ylim(10e-6,3)
    plt.minorticks_on
    plt.grid(True, which='both')
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.tick_params(axis='both', which='both', length = 5, width = 1)
    plt.title('Kappa vs Vs30 ' + 'SNR '+np.str(snr)+' '+np.str(bins)+' bins '+np.str(cut)+' sec cut', fontsize = 16)
    
    
    
    # these are matplotlib.patch.Patch properties
       
    # place a text box in upper left in axes coords
    plt.text(0.02, 5.25, textstr,  fontsize=14, verticalalignment='top',)
    
    plt.scatter(kappa, lnvs30)
    for i in range(30):
        if i == 1 or i == 10 or i ==7:
            alignment = 'right'
        else:
            alignment = 'left'
        plt.annotate(name[i], (kappa[i], lnvs30[i]), ha=alignment, alpha =.9,rotation=0)
    plt.savefig(savepath)
plot_KvsVs30(bins, cut, snr, vs30_path,full_file_path,savepath)
#%%
#spectra plots next 3 cells
    

working_dir =  '/home/eking/Documents/internship/data/Kappa/'
run_dirs = glob.glob(working_dir + 'SNR_*/*_bins/*/')


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
print(run_dirs)
for i in range(len(run_dirs)):
    kappafile = run_dirs[i] + '/kappa_site.out'
    kappaFL.append(kappafile)
    con_stn = glob.glob(run_dirs[i] + '/'+ con_dir + '/[!2]*.out')
    
    


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

colorlist = ['b','r','g','y','c','m']
run_list = ['60 sec, 75 bin, SNR 3','15 sec, 75 bin, SNR 3','15 sec, 150 bin, SNR 3','60 sec, 75 bin, SNR 5','15 sec, 75 bin, SNR 5','15 sec, 150 bin, SNR 5']
for i in range(len(station_list)):
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
    plt.title(station_str_list[i], fontsize = 16)
    for j in range(6):
        kappafile = pd.read_csv(kappaFL[j])
        stn = np.genfromtxt(station_list[i][j])
        stnid = station_str_list[i]
        ind = int(np.array(np.where(kappafile['Name'] == stnid)))
        kappa = round_sig(kappafile[' tstar(s) '][ind])
        std = round_sig(kappafile[' tstar_std '][ind])
        label = np.str(str(kappa) +' +- '+ str(std)+'; '+ run_list[j])
        freq = stn.T[0]
        amp = stn.T[1]
        std = stn.T[2]
        plt_ind = np.argmin(np.abs(freq - 35))
        
        plt.errorbar(freq[0:plt_ind] , amp[0:plt_ind], yerr=std[0:plt_ind], fmt = colorlist[j], label= label)#+ 'plus or minus'+ std)
        #plt.text(20,10**-1.1,kappa)
       
        #plt.loglog(freq, Brune_list[ind], color = 'blue', label = 'Brune spectra')
        plt.legend(loc = 'lower left', fontsize = 16)
       
        
        
       # plt.text(0.7, .1, 'Median log(diff) 1-32.7 Hz (demeaned): ' + str(round(sum_list[ind],3)), fontsize = 16)
    plt.savefig('/home/eking/Documents/internship/data/Kappa/station_spectra_2/'  + stnid + '.png')
    plt.show()
#        
    
#    outfile = open(outfile_path + '/' + stnid + '.out', 'w')
#    out = (np.array([freq_list, amp, std])).T
#    outfile.write('#freq_bins \t vel_spec_NE_m \t stdev_m \n')
#    np.savetxt(outfile, out, fmt=['%E', '%E', '%E'], delimiter='\t')
#
#    outfile.close()
##################################################################################################33
##not in log space anymore
#for i in range(len(con_ev)):#for each event
#    #make each row into an array
#    event = np.genfromtxt(con_ev[i])
#    eventid = path.basename(con_ev[i]).split('.')[0]
#    freq = event.T[0]
#    amp = event.T[1]
#    std = event.T[2]
#    fig = plt.figure(figsize = (12,10))
#    
#    plt.axes().set_xscale("log")
#    plt.axes().set_yscale("log")
#    plt.ylabel('Velocity amplitude (m)', fontsize = 16)
#    plt.xlim(0.5,70)
#    plt.errorbar(freq , amp, yerr=std, fmt = 'b')
#    plt.grid()
#    #plt.loglog(freq, Brune_list[ind], color = 'blue', label = 'Brune spectra')
#    #plt.legend(loc = 'lower right', fontsize = 16)
#    plt.xlabel('Frequency (Hz)', fontsize = 16)
#    plt.title(eventid, fontsize = 16)
#    plt.tick_params(axis='both', which='major', labelsize=15)
#    plt.tick_params(axis='both', which='both', length = 5, width = 1)
#   # plt.text(0.7, .1, 'Median log(diff) 1-32.7 Hz (demeaned): ' + str(round(sum_list[ind],3)), fontsize = 16)
#    plt.savefig(fig_dir + '/' + eventid + '.png')
#    plt.show()

#    outfile = open(outfile_path + '/' + eventid + '.out', 'w')
#    out = (np.array([freq_list, amp, std])).T
#    outfile.write('#freq_bins \t vel_spec_NE_m \t stdev_m \n')
#    np.savetxt(outfile, out, fmt=['%E', '%E', '%E'], delimiter='\t')
#    outfile.close()
#    
#%%
#kappa maps next 2 cellls    #3

def read_obj_list(objfile):
   '''
   Read a pickle file with a list of event of station objects
   Input:
       objfile:        String with path to the pickle file containing the list of objects
   Output:
       obj_list:   List with event or station objects in it
   '''
   #import pickle
   #Open the file
   obj=open(objfile,'rb')
   #Zero out a list to add the event objects to:
   obj_list=[]
   #REad the data...
   readfile=True
   while readfile==True:
       try:
           obji=pickle.load(obj)
           obj_list.append(obji)
       except EOFError:
           print ('File read complete - '+objfile)
           readfile=False
           obj.close()
   return obj_list
    
#%%
#def kappa_plot(flatfile, kappafile, fig_dir, snr, cut, bins,colors):

bins= 150
cut = 15
snr = 5

quatfaults = read_obj_list('/home/eking/Documents/internship/data/Kappa/kappa_maps/latest_quaternary_bayA.pckl')
colors = 'gist_rainbow'
kappafile = '/home/eking/Documents/internship/data/Kappa/SNR_'+np.str(snr)+'/'+np.str(bins)+'_bins/'+np.str(cut)+'/kappa_site.out'
flatfile = '/home/eking/Documents/internship/data/Kappa/SNR_'+np.str(snr)+'/SNR_'+np.str(snr)+'_file.csv'
fig_dir = '/home/eking/Documents/internship/data/Kappa/kappa_maps/'    
finalfile = '/home/eking/Documents/internship/data/Kappa/SNR_'+np.str(snr)+'/'+np.str(bins)+'_bins/'+np.str(cut)+'/full_file.csv'
kappa_file = pd.read_csv(kappafile)
flat_file = pd.read_csv(flatfile)

sflatfile = flat_file.drop_duplicates('Name')               


fullfile = pd.merge(sflatfile, kappa_file, on='Name')

print(fullfile)
fullfile.to_csv(finalfile)
#    snr = str(snr)
#    cut=str(cut)
#    bins = str(bins)
#
#slat = []
#slon = []
#sname = []
#skappa = []
minlat = 36
maxlat = 39

minlon = -124
maxlon = -120.5
#-122.35, -122.15, 37.75, 37.95
#              -124   -120.5  36  39
slat = fullfile['Slat']
  
slon = fullfile['Slon']

sname = fullfile['Name']

skappa = fullfile[' tstar(s) ']

ilist = []

for i in range(len(sname)):
    if ((minlon <= slon[i] <= maxlon) & (minlat <= slat[i] <= maxlat)):
        ilist.append(i)
print(ilist)
print(skappa)

stamen_terrain = cimgt.Stamen('terrain-background')

fig = plt.figure(figsize=(12,10))

plt.suptitle('Kappa Map '+np.str(cut)+' sec cut ' +np.str(bins)+' bins SNR '+np.str(snr), fontsize='x-large')
# Create a GeoAxes in the tile's projection.
ax = fig.add_subplot(1, 1, 1, projection=stamen_terrain.crs)


# Limit the extent of the map to a small longitude/latitude range.
ax.set_extent([minlon, maxlon, minlat, maxlat], crs=ccrs.Geodetic())

#               -122.5  -121.5  37.5 38.5
#              -124   -120.5  36  39
# Add the Stamen data at zoom level 8.
ax.add_image(stamen_terrain, 8)

transform1 = ccrs.PlateCarree()._as_mpl_transform(ax)

#ax.add_title('Station Map SNR' + SNR)
Scat = ax.scatter(slon,slat,s=120,c=skappa,marker='^', cmap=colors , alpha=1, transform=ccrs.PlateCarree())
for i_fault in range(len(quatfaults)):
    i_faultseg = quatfaults[i_fault]
    ax.plot(i_faultseg[:,0],i_faultseg[:,1],c='k',linewidth=.8, transform=ccrs.PlateCarree())
for i in ilist:
#                    #ilist
    if i != (3,25):
        ax.annotate(sname[i], (slon[i], slat[i]),xycoords=transform1, ha='left', alpha =.9)
    if i == (3,25):
        ax.annotate(sname[i], (slon[i], slat[i]),xycoords=transform1, ha='right', alpha =.9)

mapgl = ax.gridlines(draw_labels=True)

## Change the gridline labeling: https://scitools.org.uk/cartopy/docs/latest/matplotlib/gridliner.html
mapgl.xlabels_top=False
mapgl.ylabels_left=False
mapgl.xlabel_style = {'size': 12, 'color': 'black'}
mapgl.ylabel_style = {'size': 12, 'color': 'black'}

cb = plt.colorbar(Scat, format='%.3f')
cb.ax.set_ylabel('Kappa (s)', rotation=270)
cb.set_clim(0,.1)
#plt.savefig(fig_dir + '/faults/Kappa Map '+np.str(cut)+' sec cut ' +np.str(bins)+' bins SNR '+np.str(snr))
##

#%%
bins= 150
cut = 15
snr = 3
     
colors = 'gist_rainbow'
kappafile = '/home/eking/Documents/internship/data/Kappa/SNR_'+np.str(snr)+'/'+np.str(bins)+'_bins/'+np.str(cut)+'/kappa_site.out'
flatfile = '/home/eking/Documents/internship/data/Kappa/SNR_'+np.str(snr)+'/SNR_'+np.str(snr)+'_file.csv'
fig_dir = '/home/eking/Documents/internship/data/Kappa/kappa_maps/'

kappa_plot(flatfile,kappafile,fig_dir,snr,cut,bins,colors)

#%%


def plot_dataset(flatfile, fig_dir, SNR):
    
    'flatfile = path for flatfile download'
    'fig_dir = dir for saving plots'
    #PARAMS
    nstats, ustats = kappa_utils.num_stas(flatfile)# of stations with good recordings for this event'
    nevs, uevs = kappa_utils.num_evs(flatfile)# of events recorded at this station'
    flat_file = pd.read_csv(flatfile)
    eflatfile = flat_file.drop_duplicates('Quake#')
    sflatfile = flat_file.drop_duplicates('Name')               
    
    elat = eflatfile['Qlat']
    
    elon = eflatfile['Qlon']
   
    mag = eflatfile['Mag']
   
#    depth = flat_file['Qdep']
#    elev = flat_file['Selv']
#    
    slat = sflatfile['Slat']
  
    slon = sflatfile['Slon']
    bmag = flat_file['Mag']
    R = flat_file['rhyp']
    
################################################################################################################
#M_R PLOT    
    sns.set_style('darkgrid')
    fig = sns.jointplot(x=R, y=bmag)# data=flat_file)

    fig.ax_joint.set_xscale('log')

    plt.suptitle('M_R Plot SNR' + SNR )
    plt.savefig(fig_dir + 'M_R_SNR_' + SNR + '.png')
    
#################################################################################################################
    #EVENT PLOT
    stamen_terrain = cimgt.Stamen('terrain-background')

    fig = plt.figure(figsize=(10,10))
    
    plt.suptitle('Event Map SNR' + SNR )
    # Create a GeoAxes in the tile's projection.
    ax = fig.add_subplot(1, 1, 1, projection=stamen_terrain.crs)
    
    # Limit the extent of the map to a small longitude/latitude range.
    ax.set_extent([-124, -120.5, 36, 39], crs=ccrs.Geodetic())
    
    # Add the Stamen data at zoom level 8.
    ax.add_image(stamen_terrain, 8)
    #ax.add_title('Event Map SNR' + SNR)
    scat=ax.scatter(elon,elat,s=10*np.exp(mag/1.5),c=nstats,cmap='magma_r', alpha=0.7, transform=ccrs.PlateCarree())
    
    
    
    mapgl = ax.gridlines(draw_labels=True)
    
    ## Change the gridline labeling: https://scitools.org.uk/cartopy/docs/latest/matplotlib/gridliner.html
    mapgl.xlabels_top=False
    mapgl.ylabels_left=False
    mapgl.xlabel_style = {'size': 10, 'color': 'black'}
    mapgl.ylabel_style = {'size': 10, 'color': 'black'}
    
    cb = plt.colorbar(scat)
    
    plt.savefig(fig_dir + 'quakemap_SNR_' + SNR + '.png')
        
##########################################################################################################
    #STATION_PLOT
    stamen_terrain = cimgt.Stamen('terrain-background')

    fig = plt.figure(figsize=(10,10))
    
    plt.suptitle('Station Map SNR' + SNR )
    # Create a GeoAxes in the tile's projection.
    ax = fig.add_subplot(1, 1, 1, projection=stamen_terrain.crs)
    
    # Limit the extent of the map to a small longitude/latitude range.
    ax.set_extent([-124, -120.5, 36, 39], crs=ccrs.Geodetic())
    
    # Add the Stamen data at zoom level 8.
    ax.add_image(stamen_terrain, 8)
    #ax.add_title('Station Map SNR' + SNR)
    Scat=ax.scatter(slon,slat,s=100,c=nevs,marker='^', cmap=None, alpha=0.7, transform=ccrs.PlateCarree())
    
    
    
    mapgl = ax.gridlines(draw_labels=True)
    
    ## Change the gridline labeling: https://scitools.org.uk/cartopy/docs/latest/matplotlib/gridliner.html
    mapgl.xlabels_top=False
    mapgl.ylabels_left=False
    mapgl.xlabel_style = {'size': 10, 'color': 'black'}
    mapgl.ylabel_style = {'size': 10, 'color': 'black'}
    
    cb = plt.colorbar(Scat)
    
    plt.savefig(fig_dir + 'stationmap_SNR_' + SNR + '.png')
###############################################################################################################
    #BOTH PLOT
    stamen_terrain = cimgt.Stamen('terrain-background')
    
    fig = plt.figure(figsize=(10,10))
    
    plt.suptitle('Event/Station Map SNR' + SNR )
    # Create a GeoAxes in the tile's projection.
    ax = fig.add_subplot(1, 1, 1, projection=stamen_terrain.crs)
    
    # Limit the extent of the map to a small longitude/latitude range.
    ax.set_extent([-124, -120.5, 36, 39], crs=ccrs.Geodetic())
    
    # Add the Stamen data at zoom level 8.
    ax.add_image(stamen_terrain, 8)
    
    scat=ax.scatter(elon,elat,s=10*np.exp(mag/1.5),c=nstats,cmap='magma_r', alpha=0.7, transform=ccrs.PlateCarree())
    
    Scat=ax.scatter(slon,slat,s=50,c=nevs,marker='^', cmap='Blues', alpha=0.7, transform=ccrs.PlateCarree())

    for i_rec in range(len(flat_file)):
        ax.plot([flat_file['Qlon'][i_rec],flat_file['Slon'][i_rec]],[flat_file['Qlon'][i_rec],flat_file['Slon'][i_rec]],alpha=0.6, transform=ccrs.PlateCarree())
    
    mapgl = ax.gridlines(draw_labels=True)
    
    ## Change the gridline labeling: https://scitools.org.uk/cartopy/docs/latest/matplotlib/gridliner.html
    mapgl.xlabels_top=False
    mapgl.ylabels_left=False
    mapgl.xlabel_style = {'size': 10, 'color': 'black'}
    mapgl.ylabel_style = {'size': 10, 'color': 'black'}
    
    cb = plt.colorbar(scat)
    
    Cb = plt.colorbar(Scat)
    
    plt.savefig(fig_dir + 'Quake&Station_Map_SNR_' + SNR + '.pdf')

#%%
#SNRlist = [2,3,5,10]
#for i in SNRlist:
#    
#    SNR = str(i)
#    flatfile = '/home/eking/Documents/internship/data/events/SNR_'+ SNR + '/SNR_' + SNR + '_file.csv'
#    fig_dir = '/home/eking/Documents/internship/data/figs/SNR_' + SNR + '/'
#    print(len(pd.read_csv(flatfile)))
#    plot_dataset(flatfile, fig_dir, SNR)
#    
#    
    

