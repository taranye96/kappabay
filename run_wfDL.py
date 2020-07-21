#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 15:52:42 2019

@author: vjs
"""
## Runtime file for downloading waveforms


#Now make synthetics for source/station pairs
def run_parallel_dataDL(software_location,main_flatfile_path,mpi_flatfile_directory,ncpus,eventdir_unfiltered,eventdir_filtered,client_name,resp_prefilt_bottom,respfilt_bottom,start_delta,end_delta,fig_x,fig_y,output_directory):
    '''
    This routine will take the impulse response (GFs) and pass it into the routine that will
    convovle them with the source time function according to each subfaults strike and dip.
    The result fo this computation is a time series dubbed a "synthetic"
    
    IN:
        software_location:      String to location that the parallel_data.py file is in WITHOUT slash at end
        main_flatfile_path:     String to file with all possible record metadata WITHOUT slash at end
        mpi_flatfile_directory: String with directory of where to store the mpi flatfiles
        ncpus:                  Integer with number of cpus to run on
        eventdir_unfiltered:    String with the path to the unfiltered event save directory
        eventdir_filtered:      String with the path to the filtered event save directory
        client_name:            String with the client name (i.e., 'NCEDC')
        resp_prefilt_bottom:    List with the bottom two values in Hz to use in prefilt to use response (i.e., [0.005, 0.006])
        respfilt_bottom:        List with the bottom two values in Hz to use in prefilt w/ filtering as well
        start_delta:            Float with time difference in seconds to subtract from p-wave arrival for downloading waveforms
        end_delta:              Float with time difference in seconds to add to S wave arrival for downloading waveforms
        fig_x:                  Figure size x in inches to use for plotting the waveforms 
        fig_y:                  Figure size y in inches to use for plotting the waveforms 
        output_directory:       Float with path of directory to use for output flatfile csv's
        
    OUT:
        Outputs flatfiles in same format as main flatfile, but for only collected data. One for each rank (process) 
    '''
    from numpy import arange,floor,mod,append,str
    import datetime
    import subprocess
    from shlex import split
    import pandas as pd
    
    ##Time for log file
    now=datetime.datetime.now()
    now=now.strftime('%b-%d-%H%M')
    ## First read full flat file
    allmetadata=pd.read_csv(main_flatfile_path)
    ## Create individual source files
    ##   Split it up based on CPUs such that it's nearly even with 
    ##   numrecords / ncpus, but the last one will get the remainder.
    ## Get the dividing number - number of lines in all files except the last:
    n_lenfiles = floor(len(allmetadata)/ncpus)
    nmod = mod(len(allmetadata),ncpus)
    
    ## Make an array that is the start/end positions of the lines of each sub file:
    len_files = arange(0,len(allmetadata),n_lenfiles)
    
    ## If the total number of records is not evenly dividable by the ncpus,
    ##  need to replace the last number of this with the number of records (-1):
    if nmod != 0:
        len_files[-1] = len(allmetadata)
    ## If it is, then arange won't include the last number (number of records),
    ##   so append thsi manually:
    elif nmod == 0:
        len_files = append(len_files,len(allmetadata))
    
    ## Loop through the number of CPUs and grab lines to separate out files:
    for k in range(ncpus):
        ## pandas includes the last index you give it, so subtract one:
        sub_metadata = allmetadata.loc[len_files[k]:(len_files[k+1]-1)]
        sub_metadata.to_csv(mpi_flatfile_directory + '/flatfile_' + str(k) + '.csv')
        
    ## Get resp information out:
    resp_prefilt_0 = resp_prefilt_bottom[0]
    resp_prefilt_1 = resp_prefilt_bottom[1]
    respfilt_0 = resp_prefilt_bottom[0]
    respfilt_1 = resp_prefilt_bottom[1]
    
    #Make mpi system call
    print("MPI: Starting synthetics computation on", ncpus, "CPUs\n")

    ## Run on mac with --use-hwthread-cpus to force it to use all logical cpus
#    mpi='mpiexec -n '+str(ncpus)+' --use-hwthread-cpus python '+software_location+'/parallel_data.py parallel_waveformDL '+mpi_flatfile_directory+' '+eventdir_unfiltered+' '+eventdir_filtered+' '+client_name+' '+str(resp_prefilt_0)+' '+str(resp_prefilt_1)+' '+str(respfilt_0)+' '+str(respfilt_1)+' '+str(start_delta)+' '+str(end_delta)+' '+str(fig_x)+' '+str(fig_y)+' '+str(output_directory)
    ## Run on ubuntu (doesn't like thread command)
    mpi='mpiexec -n '+str(ncpus)+'  python '+software_location+'/parallel_data.py parallel_waveformDL '+mpi_flatfile_directory+' '+eventdir_unfiltered+' '+eventdir_filtered+' '+client_name+' '+str(resp_prefilt_0)+' '+str(resp_prefilt_1)+' '+str(respfilt_0)+' '+str(respfilt_1)+' '+str(start_delta)+' '+str(end_delta)+' '+str(fig_x)+' '+str(fig_y)+' '+str(output_directory)

    print(mpi)
    mpi=split(mpi)
    p=subprocess.Popen(mpi)
    p.communicate()