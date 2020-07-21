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

##%%
#kappa_utils.check_diff('Andrews_inversion_constrained', 'Andrews_inversion_constrained_np')
#
##%%
#cutlength = 15
#bins = 75
#working_dir =  '/home/eking/Documents/internship/data/Kappa/SNR_5/'+np.str(bins)+'_bins/'+np.str(cutlength)
#fig_dir = working_dir + '/figs_' + np.str(cutlength)
##fill in constraint event
#con_dir = 'Andrews_inversion_constrained_' + np.str(cutlength)
#
#kappafile = pd.read_csv(working_dir + '/kappa_site.out')
#con_ev =  glob.glob(working_dir + '/' + con_dir + '/2*.out')
#con_stn = glob.glob(working_dir + '/'+ con_dir + '/[!2]*.out')
# 
#
#
#
#  
#
###not in log space anymore
##for i in range(len(con_ev)):#for each event
##    #make each row into an array
##    event = np.genfromtxt(con_ev[i])
##    eventid = path.basename(con_ev[i]).split('.')[0]
##    freq = event.T[0]
##    amp = event.T[1]
##    std = event.T[2]
##    fig = plt.figure(figsize = (12,10))
##    
##    plt.axes().set_xscale("log")
##    plt.axes().set_yscale("log")
##    plt.ylabel('Velocity amplitude (m)', fontsize = 16)
##    plt.xlim(0.5,70)
##    plt.errorbar(freq , amp, yerr=std, fmt = 'b')
##    plt.grid()
##    #plt.loglog(freq, Brune_list[ind], color = 'blue', label = 'Brune spectra')
##    #plt.legend(loc = 'lower right', fontsize = 16)
##    plt.xlabel('Frequency (Hz)', fontsize = 16)
##    plt.title(eventid, fontsize = 16)
##    plt.tick_params(axis='both', which='major', labelsize=15)
##    plt.tick_params(axis='both', which='both', length = 5, width = 1)
##   # plt.text(0.7, .1, 'Median log(diff) 1-32.7 Hz (demeaned): ' + str(round(sum_list[ind],3)), fontsize = 16)
##    plt.savefig(fig_dir + '/' + eventid + '.png')
##    plt.show()
#
##    outfile = open(outfile_path + '/' + eventid + '.out', 'w')
##    out = (np.array([freq_list, amp, std])).T
##    outfile.write('#freq_bins \t vel_spec_NE_m \t stdev_m \n')
##    np.savetxt(outfile, out, fmt=['%E', '%E', '%E'], delimiter='\t')
##    outfile.close()
##    
#
#for i in range(len(con_stn)):#for each station
#    #make each row into an array
#    stn = np.genfromtxt(con_stn[i])
#    stnid = path.basename(con_stn[i]).split('.')[0]
#    ind = int(np.array(np.where(kappafile['site '] == stnid)))
#    kappa = kappafile[' tstar(s) '][ind]
#    freq = stn.T[0]
#    amp = stn.T[1]
#    std = stn.T[2]
#    fig = plt.figure(figsize = (10,10))
#    plt.axes().set_xscale("log")
#    plt.axes().set_yscale("log")
#    plt.ylabel('Velocity amplitude (m)', fontsize = 16)
#    plt.xlim(0.5,70)
#    plt.errorbar(freq , amp, yerr=std, fmt = 'b')
#    plt.text(20,10**-1.1,kappa)
#    plt.minorticks_on
#    plt.grid(True, which='both')
#    #plt.loglog(freq, Brune_list[ind], color = 'blue', label = 'Brune spectra')
#    #plt.legend(loc = 'lower right', fontsize = 16)
#    plt.xlabel('Frequency (Hz)', fontsize = 16)
#    plt.title(stnid, fontsize = 16)
#    plt.tick_params(axis='both', which='major', labelsize=15)
#    plt.tick_params(axis='both', which='both', length = 5, width = 1)
#   # plt.text(0.7, .1, 'Median log(diff) 1-32.7 Hz (demeaned): ' + str(round(sum_list[ind],3)), fontsize = 16)
#    plt.savefig(fig_dir + '/' + stnid + '.png')
#    plt.show()
#    
#    
##    outfile = open(outfile_path + '/' + stnid + '.out', 'w')
##    out = (np.array([freq_list, amp, std])).T
##    outfile.write('#freq_bins \t vel_spec_NE_m \t stdev_m \n')
##    np.savetxt(outfile, out, fmt=['%E', '%E', '%E'], delimiter='\t')
##
##    outfile.close()
#
##%%
#def kappa_plot(flatfile, kappafile, fig_dir, snr, cut, bins,colors):
#    
#    
#    kappa_file = pd.read_csv(kappafile)
#    flat_file = pd.read_csv(flatfile)
#    
#    sflatfile = flat_file.drop_duplicates('Name')               
#    
#    
#    fullfile = pd.merge(sflatfile, kappa_file, on='Name')
#    
#    print(fullfile)
##    snr = str(snr)
##    cut=str(cut)
##    bins = str(bins)
#    #
#    #slat = []
#    #slon = []
#    #sname = []
#    #skappa = []
#    
#    slat = fullfile['Slat']
#      
#    slon = fullfile['Slon']
#    
#    sname = fullfile['Name']
#    
#    skappa = fullfile[' tstar(s) ']
#    
#    
#    print(skappa)
#    
#    stamen_terrain = cimgt.Stamen('terrain-background')
#    
#    fig = plt.figure(figsize=(12,10))
#    
#    plt.suptitle('Kappa Map '+np.str(cut)+' sec cut ' +np.str(bins)+' bins SNR '+np.str(snr))
#    # Create a GeoAxes in the tile's projection.
#    ax = fig.add_subplot(1, 1, 1, projection=stamen_terrain.crs)
#    
#    # Limit the extent of the map to a small longitude/latitude range.
#    ax.set_extent([-124, -120.5, 36, 39], crs=ccrs.Geodetic())
#    
#    # Add the Stamen data at zoom level 8.
#    ax.add_image(stamen_terrain, 8)
#    
#    transform1 = ccrs.PlateCarree()._as_mpl_transform(ax)
#    
#    #ax.add_title('Station Map SNR' + SNR)
#    Scat = ax.scatter(slon,slat,s=120,c=skappa,marker='^', cmap=colors , alpha=1, transform=ccrs.PlateCarree())
#    for i in range(len(sname)):
#        if i != (2,25):
#            ax.annotate(sname[i], (slon[i], slat[i]),xycoords=transform1, ha='left', alpha =.7)
#        else:
#            ax.annotate(sname[i], (slon[i], slat[i]),xycoords=transform1, ha='right', alpha =.7)
#    
#    mapgl = ax.gridlines(draw_labels=True)
#    
#    ## Change the gridline labeling: https://scitools.org.uk/cartopy/docs/latest/matplotlib/gridliner.html
#    mapgl.xlabels_top=False
#    mapgl.ylabels_left=False
#    mapgl.xlabel_style = {'size': 10, 'color': 'black'}
#    mapgl.ylabel_style = {'size': 10, 'color': 'black'}
#    
#    cb = plt.colorbar(Scat, format='%.3f')
#    cb.ax.set_ylabel('Kappa (s)', rotation=270)
#    cb.set_clim(0,.1)
#    plt.savefig(fig_dir + '/uniform_colormap/Kappa Map '+np.str(cut)+' sec cut ' +np.str(bins)+' bins SNR '+np.str(snr))
#    ##
#
##%%
#bins= 75
#cut = 60
#snr = 3
#     
#colors = 'gist_rainbow'
#kappafile = '/home/eking/Documents/internship/data/Kappa/SNR_'+np.str(snr)+'/'+np.str(bins)+'_bins/'+np.str(cut)+'/kappa_site.out'
#flatfile = '/home/eking/Documents/internship/data/Kappa/SNR_'+np.str(snr)+'/SNR_'+np.str(snr)+'_file.csv'
#fig_dir = '/home/eking/Documents/internship/data/Kappa/kappa_maps/'
#
#kappa_plot(flatfile,kappafile,fig_dir,snr,cut,bins,colors)
#
##%%
#

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
#    sns.set_style('darkgrid')
#    fig = sns.jointplot(x=R, y=bmag)# data=flat_file)
#
#    fig.ax_joint.set_xscale('log')
#
#    plt.suptitle('M_R Plot SNR' + SNR )
#    plt.savefig(fig_dir + 'M_R_SNR_' + SNR + '.png')
#    
#################################################################################################################
#    #EVENT PLOT
#    stamen_terrain = cimgt.Stamen('terrain-background')
#
#    fig = plt.figure(figsize=(10,10))
#    
#    plt.suptitle('Event Map SNR' + SNR )
#    # Create a GeoAxes in the tile's projection.
#    ax = fig.add_subplot(1, 1, 1, projection=stamen_terrain.crs)
#    
#    # Limit the extent of the map to a small longitude/latitude range.
#    ax.set_extent([-124, -120.5, 36, 39], crs=ccrs.Geodetic())
#    
#    # Add the Stamen data at zoom level 8.
#    ax.add_image(stamen_terrain, 8)
#    #ax.add_title('Event Map SNR' + SNR)
#    scat=ax.scatter(elon,elat,s=10*np.exp(mag/1.5),c=nstats,cmap='magma_r', alpha=0.7, transform=ccrs.PlateCarree())
#    
#    
#    
#    mapgl = ax.gridlines(draw_labels=True)
#    
#    ## Change the gridline labeling: https://scitools.org.uk/cartopy/docs/latest/matplotlib/gridliner.html
#    mapgl.xlabels_top=False
#    mapgl.ylabels_left=False
#    mapgl.xlabel_style = {'size': 10, 'color': 'black'}
#    mapgl.ylabel_style = {'size': 10, 'color': 'black'}
#    
#    cb = plt.colorbar(scat)
#    
#    plt.savefig(fig_dir + 'quakemap_SNR_' + SNR + '.png')
#        
##########################################################################################################
#    #STATION_PLOT
#    stamen_terrain = cimgt.Stamen('terrain-background')
#
#    fig = plt.figure(figsize=(10,10))
#    
#    plt.suptitle('Station Map SNR' + SNR )
#    # Create a GeoAxes in the tile's projection.
#    ax = fig.add_subplot(1, 1, 1, projection=stamen_terrain.crs)
#    
#    # Limit the extent of the map to a small longitude/latitude range.
#    ax.set_extent([-124, -120.5, 36, 39], crs=ccrs.Geodetic())
#    
#    # Add the Stamen data at zoom level 8.
#    ax.add_image(stamen_terrain, 8)
#    #ax.add_title('Station Map SNR' + SNR)
#    Scat=ax.scatter(slon,slat,s=100,c=nevs,marker='^', cmap=None, alpha=0.7, transform=ccrs.PlateCarree())
#    
#    
#    
#    mapgl = ax.gridlines(draw_labels=True)
#    
#    ## Change the gridline labeling: https://scitools.org.uk/cartopy/docs/latest/matplotlib/gridliner.html
#    mapgl.xlabels_top=False
#    mapgl.ylabels_left=False
#    mapgl.xlabel_style = {'size': 10, 'color': 'black'}
#    mapgl.ylabel_style = {'size': 10, 'color': 'black'}
#    
#    cb = plt.colorbar(Scat)
#    
#    plt.savefig(fig_dir + 'stationmap_SNR_' + SNR + '.png')
###############################################################################################################
    #BOTH PLOT
    stamen_terrain = cimgt.Stamen('terrain-background')
    
    fig = plt.figure(figsize=(12,10))
    
    plt.suptitle('Event/Station Map SNR' + SNR )
    # Create a GeoAxes in the tile's projection.
    ax = fig.add_subplot(1, 1, 1, projection=stamen_terrain.crs)
    
    # Limit the extent of the map to a small longitude/latitude range.
    ax.set_extent([-124, -120.5, 36, 39], crs=ccrs.Geodetic())
    
    # Add the Stamen data at zoom level 8.
    ax.add_image(stamen_terrain, 8)
    
    scat=ax.scatter(elon,elat,s=10*np.exp(mag/1.5),c=nstats,cmap='magma_r', alpha=0.6, transform=ccrs.PlateCarree())
    
    Scat=ax.scatter(slon,slat,s=80,c=nevs,marker='^', cmap='Blues', alpha=1, edgecolors='k', transform=ccrs.PlateCarree())

    for i_rec in range(len(flat_file)):
        ax.plot([flat_file['Qlon'][i_rec],flat_file['Slon'][i_rec]],[flat_file['Qlon'][i_rec],flat_file['Slon'][i_rec]],alpha=0.6, transform=ccrs.PlateCarree())
    
    mapgl = ax.gridlines(draw_labels=True)
    
    ## Change the gridline labeling: https://scitools.org.uk/cartopy/docs/latest/matplotlib/gridliner.html
    mapgl.xlabels_top=False
    mapgl.ylabels_left=True
    mapgl.ylabels_right=False
    mapgl.xlabel_style = {'size': 12, 'color': 'black'}
    mapgl.ylabel_style = {'size': 12, 'color': 'black'}
    

    
    Cb = plt.colorbar(Scat, fraction=0.046, pad=0.04)
    
   
    
    cb = plt.colorbar(scat, fraction=0.046, pad=0.04)
    
  
    
    plt.savefig(fig_dir + 'Quake&Station_Map_SNR_' + SNR + '.pdf')

#%%
SNRlist = [3,5]
for i in SNRlist:
    
    SNR = str(i)
    flatfile = '/home/eking/Documents/internship/data/events/SNR_'+ SNR + '/SNR_' + SNR + '_file.csv'
    fig_dir = '/home/eking/Documents/internship/data/figs/SNR_' + SNR + '/'
    print(len(pd.read_csv(flatfile)))
    plot_dataset(flatfile, fig_dir, SNR)
    
    
    

