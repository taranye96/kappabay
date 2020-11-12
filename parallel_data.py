#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 15:57:35 2019

@author: eking
"""


def parallel_waveformDL(mpi_flatfile_directory,eventdir_unfiltered,eventdir_filtered,client_name,resp_prefilt_bottom,respfilt_bottom,start_delta,end_delta,fig_x,fig_y,output_directory,rank,size):
    '''
    Run the waveform data download in parallel.
    Attempts to download waveforms for record information in each line of 
    latfile_csvpath. Removes instrument response and saves as unfiltered, also
    removes inst. resp and filters and saves in a separate directory for the 
    event, directory structure supplied as input. Data and plots only saved if
    SNR >=5.  Makes a figure of the waveform with theoretical P and S wave
    arrival times, and outputs new flatfile.
    Input:
        mpi_flatfile_directory: String with the directory containing the MPI flatfiles, saved in format "flatfile_{rank}.csv"
        eventdir_unfiltered:    String with the path to the unfiltered event save directory
        eventdir_filtered:      String with the path to the filtered event save directory
        client_name:            String with the client name (i.e., 'NCEDC')
        resp_prefilt_bottom:    List with the bottom two values in Hz to use in prefilt to use response (i.e., [0.005, 0.006])
        respfilt_bottom:        List with the bottom two values in Hz to use in prefilt w/ filtering as well
        start_delta:            Float with time difference in seconds to subtract from p-wave arrival for downloading waveforms
        end_delta:              Float with time difference in seconds to add to S wave arrival for downloading waveforms
        fig_x:                  Figure size x in inches to use for plotting the waveforms 
        fig_y:                  Figure size y in inches to use for plotting the waveforms 
        output_directory:       Float with path of directory to use for output flatfile csv's, WITHOUT slash at end
        rank:                   Float with the rank
        size:                   Float with the size
    Output:
        
    '''
    
    from obspy.clients.fdsn import Client
    from obspy.core import Stream, read, UTCDateTime
    from datetime import datetime, timedelta
    import matplotlib.pyplot as plt
    from glob import glob
    import numpy as np
    import csv
    import pandas as pd
    from os import path,makedirs
    import kappa_utils as kutils
   
    data_dir1 = eventdir_unfiltered
    data_dir2 = eventdir_filtered
    flatfile_path = mpi_flatfile_directory + '/flatfile_' + np.str(rank) + '.csv'
    client = Client(client_name)
    start_td = start_delta  ## Time difference to subtract from p-wave arrival for downloading waveforms
    end_td = end_delta ## Time difference to add to S wave arrival for downloading waveforms
    ## Filter bottoms to use in prefilt for -
    ##  INstrument response ONLY:
    resp_only_filtbottom = resp_prefilt_bottom
    ##  Instrument response and filtering
    resp_filt_filtbottom = respfilt_bottom
    ## Figure size:
    figure_size = (fig_x,fig_y)
    
    
    
    ## Read in metadata:
    allmetadata = pd.read_csv(flatfile_path)
    
    ## Go through the metadata lines, extract record metadata, download waveforms,
    ##  correct instrument response / filter, make plots + save, save as a SAC in 
    ##  the appropriate directory.
    
    ## Start a counter for hte number of lines
    count = 0 
    ## Counter for the number of station/event permutations missed (no data to DL)
    nummissed = 0
    ## Empty list for... ?
    ffl = []
    
    ## Empty arrays for output flatfiles if the data was downloaded
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

    
    ## Loop over the lines in the given file 
    for i_line in range(len(allmetadata)):
        
        #if nummissed == 10000:
        #    break
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
        
        i_parr,_ = allmetadata.loc[i_line]['Parr'].split('.')
        i_sarr,_ =  allmetadata.loc[i_line]['Sarr'].split('.')
        i_network = allmetadata.loc[i_line]['Network']
        i_Parr = datetime.strptime(i_parr,"%Y-%m-%d %H:%M:%S")
        i_Sarr =datetime.strptime(i_sarr,"%Y-%m-%d %H:%M:%S")
        i_station = allmetadata.loc[i_line]['Name']
        i_stlon = allmetadata.loc[i_line]['Slon']
        i_stlat = allmetadata.loc[i_line]['Slat']
        i_stelv = allmetadata.loc[i_line]['Selv']
        
        i_rhyp = allmetadata.loc[i_line]['rhyp']
        
        sp = i_Sarr - i_Parr   ## Get the s arrival - p arrival time difference
        i_start = i_Sarr - timedelta(seconds=start_td)
        i_end = i_start + timedelta(seconds=end_td)
        i_network = str(i_network)
        i_station = str(i_station)
        
        event = 'Event'+'_'+i_year+'_'+i_month+'_'+i_day+'_'+i_hr+'_'+i_min+'_'+i_sec
        
        
    ########################################################################################################################################3
        ## Initiate empty stream obmects for the N and E channels
        raw_stn = Stream()
        raw_ste = Stream()
        
        ## Try to download data for the N channel - if it works, add it to the N 
        ##  stream object
        try:
            raw_stn += client.get_waveforms(i_network, i_station, "*", 'HHN', UTCDateTime(i_start), UTCDateTime(i_end), attach_response=True)
        ## If it's missed, add 0.5 to the missed number (half a channel missing),
        ##  and continue...
        except:
            nummissed += .5
            print('missed DL record for network ' + i_network + ', station ' + i_station + ' on channel HHN, event ' + np.str(i_eventnum))
            ## continue to next line of loop (i_line)
            continue
        
        ## Make copies of the raw station objects to use for removing instrument 
        ##   response ONLY, and for removing ins. resp and filtering:
        ir_stn = raw_stn.copy()
        ir_filt_stn = raw_stn.copy()
        
        ## Get sampling rate and make the filters to use in removing instrument response
        samprate = ir_stn[0].stats['sampling_rate']
        ## Make the prefilt for the instrment response - 1/2 sampling rate is nyquist
        prefilt1 = (resp_only_filtbottom[0], resp_only_filtbottom[1], ((samprate/2)-5), (samprate/2))  ## this is 20 to 25 at the end
        try:
            ir_stn[0].remove_response(output='VEL',pre_filt=prefilt1) ## The units of data are now Velocity, m/s
        except:
            nummissed += .5
            print('missed remove IR record for network ' + i_network + ', station ' + i_station + ' on channel HHN, event ' + np.str(i_eventnum))
            ## continue to next line of loop (i_line)
            continue
        prefilt2 = (resp_filt_filtbottom[0],resp_filt_filtbottom[1], ((samprate/2)-5), (samprate/2)) ## 0.063 just above 16 sec microsiesem, .28 just above 4 sec
        try:
            ir_filt_stn[0].remove_response(output='VEL',pre_filt=prefilt2) ## The units of data are now Velocity, m/s
        except:
            nummissed += .5
            print('missed remove IR record for network ' + i_network + ', station ' + i_station + ' on channel HHN, event ' + np.str(i_eventnum))
            ## continue to next line of loop (i_line)
            continue
        
        
    ######################################################################################################################################################################################
        ## Try to download for the E channel - if it works, do same as above
        try:
            raw_ste += client.get_waveforms(i_network, i_station, "*", 'HHE', UTCDateTime(i_start), UTCDateTime(i_end), attach_response=True)
        except:
            nummissed += .5
            print('missed DL record for network ' + i_network + ', station ' + i_station + ' on channel HHE, event ' + np.str(i_eventnum))
            continue
        ir_ste = raw_ste.copy()
        ir_filt_ste = raw_ste.copy()
        
        
        samprate = ir_ste[0].stats['sampling_rate']
        ## Make the prefilt for the instrment response - AT.SIT is @ 50Hz so 25 is nyquist
        prefilt1 = (resp_only_filtbottom[0], resp_only_filtbottom[1], ((samprate/2)-5), (samprate/2))  ## this is 20 to 25 at the end
        try:
            ir_ste[0].remove_response(output='VEL',pre_filt=prefilt1) ## The units of data are now Velocity, m/s
        except:
            nummissed += .5
            print('missed remove IR record for network ' + i_network + ', station ' + i_station + ' on channel HHE, event ' + np.str(i_eventnum))
            ## continue to next line of loop (i_line)
            continue
        prefilt2 = (resp_filt_filtbottom[0],resp_filt_filtbottom[1], ((samprate/2)-5), (samprate/2)) ## 0.063 just above 16 sec microsiesem, .28 just above 4 sec
        try:
            ir_filt_ste[0].remove_response(output='VEL',pre_filt=prefilt2) ## The units of data are now Velocity, m/s
        except:
            nummissed += .5
            print('missed remove IR record for network ' + i_network + ', station ' + i_station + ' on channel HHE, event ' + np.str(i_eventnum))
            ## continue to next line of loop (i_line)
            continue
        
        
    ######################################################################################################################################################################################
        ## Calculate SNR and only save waveforms and plots if it's >= 5.
        SNR_N = kutils.comp_SNR(ir_filt_stn, 10, 30, 15)
        SNR_E = kutils.comp_SNR(ir_filt_ste, 10, 30, 15)
        SNR_avg = (SNR_N + SNR_E)/2
        
        print(f'SNR for {i_station} {i_eventnum} is {SNR_avg}') 

        if SNR_avg >=5:
            
            ## Make sure paths exist
            if not path.exists(data_dir1+event):
                makedirs(data_dir1+event)
            if not path.exists(data_dir2+event):
                makedirs(data_dir2+event)
            
            ### North component 
            ## make and save plot of unfiltered data
            plt.figure(figsize=figure_size)
            plt.plot(ir_stn[0].times(),ir_stn[0].data,'g')
            ## Plot a vertical line for the p-wave arrival
            plt.axvline(x=start_td)
            ## Plot a vertical line for the s-wave arrival
            plt.axvline(x=start_td-sp.seconds)
            plt.xlabel('Time from ')
            plt.ylabel('Velocity (m/s)')
            plt.title('Instrument Response Removed, Unfiltered, \n' + i_network + i_station )
            plt.savefig(data_dir1+event+'/'+i_network+'_'+i_station+'_'+'HHN'+'_'+i_year+'_'+i_month+'_'+i_day+'_'+i_hr+'_'+i_min+ '_' + i_sec +'.png')
            plt.close('all')
            ir_stn[0].write(data_dir1+event+'/'+i_network+'_'+i_station+'_'+'HHN'+'_'+i_year+'_'+i_month+'_'+i_day+'_'+i_hr+'_'+i_min+ '_'+ i_sec + '.sac',format='SAC')
            #ffl.append(ir_stn[0])
            
            ## make and save plot of filtered data
            plt.figure(figsize=figure_size)
            plt.plot(ir_filt_stn[0].times(),ir_filt_stn[0].data,'g')
            ## Plot a vertical line for the p-wave arrival
            plt.axvline(x=start_td)
            ## Plot a vertical line for the s-wave arrival
            plt.axvline(x=start_td-sp.seconds)
            plt.xlabel('Time from ')
            plt.ylabel('Velocity (m/s)')
            plt.title('Instrument Response Removed, filtered, \n' + i_network +i_station )
            plt.savefig(data_dir2+event+'/'+i_network+'_'+i_station+'_'+'HHN'+'_'+i_year+'_'+i_month+'_'+i_day+'_'+i_hr+'_'+i_min+ '_' + i_sec +'.png')
            plt.close('all')
            ir_filt_stn[0].write(data_dir2+event+'/'+i_network+'_'+i_station+'_'+'HHN'+'_'+i_year+'_'+i_month+'_'+i_day+'_'+i_hr+'_'+i_min+ '_' + i_sec + '.sac',format='SAC')
            
            ### East component 
            ## make and save plot of unfiltered data
            plt.figure(figsize=figure_size)
            plt.plot(ir_ste[0].times(),ir_ste[0].data,'g')
            ## Plot a vertical line for the p-wave arrival
            plt.axvline(x=start_td)
            ## Plot a vertical line for the s-wave arrival
            plt.axvline(x=start_td-sp.seconds)
            plt.xlabel('Time from ')
            plt.ylabel('Velocity (m/s)')
            plt.title('Instrument Response Removed, Unfiltered, \n' + i_network + i_station )
            plt.savefig(data_dir1+event+'/'+i_network+'_'+i_station+'_'+'HHE'+'_'+i_year+'_'+i_month+'_'+i_day+'_'+i_hr+'_'+i_min+ '_' + i_sec +'.png')
            plt.close('all')
            ir_ste[0].write(data_dir1+event+'/'+i_network+'_'+i_station+'_'+'HHE'+'_'+i_year+'_'+i_month+'_'+i_day+'_'+i_hr+'_'+i_min+ '_' + i_sec +'.sac',format='SAC')
            #ffl.append(ir_stn[0])
            
            ## make and save plot of filtered data
            plt.figure(figsize=figure_size)
            plt.plot(ir_filt_ste[0].times(),ir_filt_ste[0].data,'g')
            ## Plot a vertical line for the p-wave arrival
            plt.axvline(x=start_td)
            ## Plot a vertical line for the s-wave arrival
            plt.axvline(x=start_td-sp.seconds)
            plt.xlabel('Time from ')
            plt.ylabel('Velocity (m/s)')
            plt.title('Instrument Response Removed, filtered, \n' + i_network + i_station )
            plt.savefig(data_dir2+event+'/'+i_network+'_'+i_station+'_'+'HHE'+'_'+i_year+'_'+i_month+'_'+i_day+'_'+i_hr+'_'+i_min+ '_' + i_sec +'.png')
            plt.close('all')
            ir_filt_ste[0].write(data_dir2+event+'/'+i_network+'_'+i_station+'_'+'HHE'+'_'+i_year+'_'+i_month+'_'+i_day+'_'+i_hr+'_'+i_min+ '_' + i_sec +'.sac',format='SAC')
            
            ## Add to arrays for new output file:
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
            out_orgt = np.append(out_orgt,i_origintime)
            out_rhyp = np.append(out_rhyp,i_rhyp)
            out_parr = np.append(out_parr,i_parr)
            out_sarr = np.append(out_sarr,i_sarr)
    
    ## Make new dataframe and flatfile with output arrays:
    ## Dict:
    collected_dict = {'Network':out_network,'Name':out_station,'Slat':out_stlat,'Slon':out_stlon,
                      'Selv':out_stelv,'Quake#':out_quakenum,'Mag':out_m,'Qlat':out_qlat,
                      'Qlon':out_qlon,'Qdep':out_qdep,'OrgT':out_orgt,'rhyp':out_rhyp,
                      'Parr':out_parr,'Sarr':out_sarr}
    collected_df = pd.DataFrame(collected_dict)
    ## Write to file:
    collected_df.to_csv(output_directory+'collected_flatfile+'+np.str(rank)+'.csv')
    
    
    ## Print out the total number of missed recordings:
    print(nummissed)
  
    
#    print(i_line)
#    count +=1
#    if count == 10:
#        break




    #If main entry point
if __name__ == '__main__':
    import sys
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print('I am rank %d of %d, hi.' %(rank,size))
    # Map command line arguments to function arguments.
    if sys.argv[1]=='parallel_waveformDL':
        #Parse command line arguments
        mpi_flatfile_directory=sys.argv[2]
        eventdir_unfiltered=sys.argv[3]
        eventdir_filtered=sys.argv[4]
        client_name=sys.argv[5]
        resp_prefilt_bottom0=float(sys.argv[6])
        resp_prefilt_bottom1=float(sys.argv[7])
        respfilt_bottom0=float(sys.argv[8])
        respfilt_bottom1=float(sys.argv[9])
        start_delta=float(sys.argv[10])
        end_delta=float(sys.argv[11])
        fig_x=float(sys.argv[12])
        fig_y=float(sys.argv[13])
        output_directory=sys.argv[14]

        parallel_waveformDL(mpi_flatfile_directory,eventdir_unfiltered,eventdir_filtered,client_name,[resp_prefilt_bottom0,resp_prefilt_bottom0],[respfilt_bottom0,respfilt_bottom1],start_delta,end_delta,fig_x,fig_y,output_directory,rank,size)
    
