#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 18:22:27 2019

@author: vjs
"""

## Run data download in parallel ## 

import parallel_data
import run_wfDL


############   PARAMETERS   ##############
#/home/eking/Documents/internship/datanew/events/
####  PATHS  ####
## Path to the larger flatfile:
#main_flatfile_path = '/home/tnye/kappa/data/flatfiles/main_flatfile.csv'
main_flatfile_path = '/home/tnye/kappa/data/flatfiles/SNR_5_file.csv'
## Path to the mpi flatfile directory
mpi_flatfile_directory = '/home/tnye/kappa/data/flatfiles/mpi_flatfiles'
## Paths to where the unfiltered and filtered .sac and .png's are saved
eventdir_unfiltered = '/home/tnye/kappa/data/waveforms/acc/unfiltered/'
eventdir_filtered = '/home/tnye/kappa/data/waveforms/acc/filtered/'
## path to where scripts to run are stored:
software_location = '/home/tnye/kappa/kappabay'

####  PARAMS  ####
## Name of the download client
client_name = 'NCEDC'
##  Instrument response ONLY:
resp_prefilt_bottom = [0.005,0.006]
##  Instrument response and filtering
respfilt_bottom = [0.063,0.28]  # 0.063 just above 16 sec microsiesem, .28 just above 4 sec
start_delta = 30  ## Time difference to subtract from p-wave arrival for downloading waveforms
end_delta = 120    ## Time difference to add to S wave arrival for downloading waveforms
## Figure size:
wf_figure_size = (12,6)
## Output directory for the collected flatfiles
output_directory =  '/home/tnye/kappa/data/flatfiles/collected_mpi_flatfiles/'
## Number of CPUs to run on
ncpus=16

###########################################

## RUN IT!! Hold your breath... 
run_wfDL.run_parallel_dataDL(software_location,main_flatfile_path,mpi_flatfile_directory,ncpus,eventdir_unfiltered,eventdir_filtered,client_name,resp_prefilt_bottom,respfilt_bottom,start_delta,end_delta,wf_figure_size[0],wf_figure_size[1],output_directory)
