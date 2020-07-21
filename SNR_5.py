#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 09:46:05 2019

@author: eking
"""

top_dir = '/home/eking/Documents/internship/data/events/'
flat_file_dir = '/home/eking/Documents/internship/data/collected_mpi_flatfiles/PhatPhile.csv'
snr_pass_dir = top_dir + 'SNR_10/'
SNR_pass = 10
noisedur = 10
sigdur = 15

def SNR_BigBoy(top_dir,flat_file_dir,snr_pass_dir,SNR_pass,noisedur,sigdur):
    
    
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
    import os.path as path
    import os
    import obspy
    from obspy import read
    import glob
    import seaborn as sns; sns.set(style="white", color_codes=True)
    import kappa_utils
    import shutil as sh
    
    

    
    #download data from flatfile
    
    allmetadata = pd.read_csv(flat_file_dir)
    
 #  somemetadata = allmetadata[:5]
    
        # arrays for output flatfile
        
    out_network = np.array([])
    out_station = np.array([])
    out_stlat = np.array([])
    out_stlon = np.array([])
    out_stelv = np.array([])
    out_quakenum = np.array([])
    out_m = np.array([])
    out_qlat = np.array([])
    out_qlon = np.array([])
    out_qdep = np.array([])
    out_orgt = np.array([])
    out_rhyp = np.array([])
    out_parr = np.array([])
    out_sarr = np.array([])
    out_SNR = np.array([])
    
    
    
    
       # big loop 
    for i_line in range(len(allmetadata)):
            
           ## get the data out of each line in the file
           
        ## Grab the appropriate metadata for this line
        i_eventnum = allmetadata.loc[i_line]['Quake#']
        i_qlon = allmetadata.loc[i_line]['Qlon']
        i_qlat = allmetadata.loc[i_line]['Qlat']
        i_qdep = allmetadata.loc[i_line]['Qdep']
        i_m = allmetadata.loc[i_line]['Mag']
        
        ##Origin time
        i_origintime = allmetadata.loc[i_line]['OrgT']
        i_date,i_time = allmetadata.loc[i_line]['OrgT'].split(' ')
        i_year,i_month,i_day = i_date.split('-')
        i_hr,i_min,i_sec = i_time.split(':')
        try:
            i_sec,_ = i_sec.split('.')
        except:
            continue
    
        i_parr = allmetadata.loc[i_line]['Parr']
        i_sarr =  allmetadata.loc[i_line]['Sarr']
        i_network = allmetadata.loc[i_line]['Network']
        try:
            i_origintime,_ = i_origintime.split('.')
        except:
            continue
        i_Otime = datetime.strptime(i_origintime,"%Y-%m-%d %H:%M:%S")
        i_Parr = datetime.strptime(i_parr,"%Y-%m-%d %H:%M:%S")
        i_Sarr =datetime.strptime(i_sarr,"%Y-%m-%d %H:%M:%S")
        i_station = allmetadata.loc[i_line]['Name']
        i_stlon = allmetadata.loc[i_line]['Slon']
        i_stlat = allmetadata.loc[i_line]['Slat']
        i_stelv = allmetadata.loc[i_line]['Selv']
        
        i_rhyp = allmetadata.loc[i_line]['rhyp']
        
#        if (i_m <= 3 and i_rhyp >=100):
#            continue
#        if i_rhyp >= 200 :
#            continue
        
            
        sigstart = 30
        
       #i_network = str(i_network)
       #i_station = str(i_station)
        
        # east and nort files
        
        i_filepath_N = top_dir +'filtered/Event_'+ i_year + '_' + i_month + '_' + i_day + '_' + i_hr + '_' + i_min +'_' + i_sec + '/' + i_network + '_' + i_station + '_HHN_' + i_year + '_' + i_month + '_' + i_day + '_' + i_hr + '_' + i_min +'_' + i_sec + '.sac'
        
        i_filepath_E = top_dir +'filtered/Event_'+ i_year + '_' + i_month + '_' + i_day + '_' + i_hr + '_' + i_min +'_' + i_sec + '/' + i_network + '_' + i_station + '_HHE_' + i_year + '_' + i_month + '_' + i_day + '_' + i_hr + '_' + i_min +'_' + i_sec + '.sac'
        
        # read files
        
        i_sac_N = obspy.read(i_filepath_N)
        i_sac_E = obspy.read(i_filepath_E)
        
        # compute SNR
        
        i_SNR_N = kappa_utils.comp_SNR(i_sac_N, noisedur, sigstart,sigdur)
        i_SNR_E = kappa_utils.comp_SNR(i_sac_E, noisedur,sigstart,sigdur)
        i_SNR_ave = (i_SNR_N + i_SNR_E)/2
        print(i_SNR_ave)
        #check SNR
        if i_SNR_ave >= SNR_pass:
            
           ## save for flat file
            
            out_network = np.append(out_network,i_network)
            out_station = np.append(out_station,i_station)
            out_stlat = np.append(out_stlat,i_stlat)
            out_stlon = np.append(out_stlon,i_stlon)
            out_stelv = np.append(out_stelv,i_stelv)
            out_quakenum = np.append(out_quakenum,i_eventnum)
            out_m = np.append(out_m,i_m)
            out_qlat = np.append(out_qlat,i_qlat)
            out_qlon = np.append(out_qlon,i_qlon)
            out_qdep = np.append(out_qdep,i_qdep)
            out_orgt = np.append(out_orgt,i_Otime)
            out_rhyp = np.append(out_rhyp,i_rhyp)
            out_parr = np.append(out_parr,i_Parr)
            out_sarr = np.append(out_sarr,i_Sarr)
            out_SNR =  np.append(out_SNR, (i_SNR_N + i_SNR_E)/2)
            
            ## save sac files and move plots over 
            
            new_SAC_filepath_N =  top_dir +'SNR_10/waveforms/Event_'+ i_year + '_' + i_month + '_' + i_day + '_' + i_hr + '_' + i_min +'_' + i_sec + '/' + i_network + '_' + i_station + '_HHN_' + i_year + '_' + i_month + '_' + i_day + '_' + i_hr + '_' + i_min +'_' + i_sec + '.sac'
        
            new_SAC_filepath_E =  top_dir +'SNR_10/waveforms/Event_'+ i_year + '_' + i_month + '_' + i_day + '_' + i_hr + '_' + i_min +'_' + i_sec + '/' + i_network + '_' + i_station + '_HHE_' + i_year + '_' + i_month + '_' + i_day + '_' + i_hr + '_' + i_min +'_' + i_sec + '.sac'
        
            i_sac_N.write(new_SAC_filepath_N, format='SAC')
            i_sac_E.write(new_SAC_filepath_E, format='SAC')
           
            fig_path_N = top_dir +'filtered/Event_'+ i_year + '_' + i_month + '_' + i_day + '_' + i_hr + '_' + i_min +'_' + i_sec + '/' + i_network + '_' + i_station + '_HHN_' + i_year + '_' + i_month + '_' + i_day + '_' + i_hr + '_' + i_min+'_' + i_sec +'.png'
            fig_path_E = top_dir +'filtered/Event_'+ i_year + '_' + i_month + '_' + i_day + '_' + i_hr + '_' + i_min +'_' + i_sec + '/' + i_network + '_' + i_station + '_HHE_' + i_year + '_' + i_month + '_' + i_day + '_' + i_hr + '_' + i_min +'_' + i_sec + '.png'
            fig_save_N = top_dir +'SNR_10/waveforms/Event_'+ i_year + '_' + i_month + '_' + i_day + '_' + i_hr + '_' + i_min +'_' + i_sec + '/' + i_network + '_' + i_station + '_HHN_' + i_year + '_' + i_month + '_' + i_day + '_' + i_hr + '_' + i_min +'_' + i_sec +'.png'
            fig_save_E = top_dir +'SNR_10/waveforms/Event_'+ i_year + '_' + i_month + '_' + i_day + '_' + i_hr + '_' + i_min +'_' + i_sec + '/' + i_network + '_' + i_station + '_HHE_' + i_year + '_' + i_month + '_' + i_day + '_' + i_hr + '_' + i_min +'_' + i_sec +'.png'
            
            sh.copy(fig_path_N,fig_save_N)
            sh.copy(fig_path_E,fig_save_E)
            
            
        else:
            continue
        # make big flat file
    collected_dict = {'Network':out_network,'Name':out_station,'Slat':out_stlat,'Slon':out_stlon,
                          'Selv':out_stelv,'Quake#':out_quakenum,'Mag':out_m,'Qlat':out_qlat,
                          'Qlon':out_qlon,'Qdep':out_qdep,'OrgT':out_orgt,'rhyp':out_rhyp,
                          'Parr':out_parr,'Sarr':out_sarr,'SNR':out_SNR}
    collected_df = pd.DataFrame(collected_dict)
        ## Write to file:
    collected_df.to_csv(snr_pass_dir+'SNR_10_file'+'.csv')
        
         
SNR_BigBoy(top_dir,flat_file_dir,snr_pass_dir,SNR_pass,noisedur,sigdur)
#%%




#%%

#def comp_SNR(stream, noisestart, noisedur, sigstart, sigdur):
#    time = stream[0].times()
#    data = stream[0].data()
#    noiseend = noisestart + noisedur
#    sigend = sigstart + sigdur
#    # find indecies
#    noise_ind = np.where(time<=noiseend) and np.where(time>=noisestart)
#    noise_amp = data[noise_ind]
#    noise_avg = np.mean(noise_amp)
#    
#    sig_ind = np.where(time<=sigend) and np.where(time>=sigstart)
#    sig_amp = data[sig_ind]
#    sig_avg = np.mean(sig_amp)
#    
#    SNR=sig_avg/noise_avg
#    
#    return SNR
#    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
