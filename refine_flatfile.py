#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 16:17:41 2018

@author: vjs
"""

#####                       Make refined flatfile                        ######
##### Take overall flatfile and separate out into the events of interest ######
#####  GEt Vs30, PGD Magnitude, and GMPE predictions for each data point...


import numpy as np
import pandas as pd
import subprocess
from shlex import split
import os
from shlex import split
import tsueqs_main_fns as tmf

## Open quake stuff:
from openquake.hazardlib import imt, const
from openquake.hazardlib.gsim.base import RuptureContext
from openquake.hazardlib.gsim.base import DistancesContext
from openquake.hazardlib.gsim.base import SitesContext
from openquake.hazardlib.gsim.zhao_2006 import ZhaoEtAl2006SInter
from openquake.hazardlib.gsim.boore_2014 import BooreEtAl2014
from openquake.hazardlib.gsim.travasarou_2003 import TravasarouEtAl2003
 
 
##########################
########## Paths #########
##########################

threshold0_path = '/Users/vjs/tsueqs/flatfiles/threshold_0cm_ia_cav.csv'
threshold1_path = '/Users/vjs/tsueqs/flatfiles/threshold_0cm_ia_cav.csv'
threshold5_path = '/Users/vjs/tsueqs/flatfiles/threshold_0cm_ia_cav.csv'
threshold10_path = '/Users/vjs/tsueqs/flatfiles/threshold_0cm_ia_cav.csv'

## Vs30 global proxy grd file to use to extract vs30 values
vs30grdfile = '/home/eking/Documents/internship/data/Kappa/global_vs30.grd'

## Path where gmt is stored to use system call
gmt_path = '/usr/bin/'

## CSV file with PGD magnitudes for the events
#pgd_mag_file = '/Users/vjs/tsueqs/data/mpgd_events.csv'

## Output path:
#refinedflatfile_path = '/Users/vjs/tsueqs/flatfiles/refined_subset.csv'

## output path for reference gpmes:
#refgmpe_path = '/Users/vjs/tsueqs/flatfiles/reference_gmpes.csv'

bins= 150
cut = 15
snr = 5

full_file_path =  '/home/eking/Documents/internship/data/Kappa/SNR_'+np.str(snr)+'/'+np.str(bins)+'_bins/'+np.str(cut)+'/full_file.csv'
##########################
########## Param #########
##########################
fixed_M = np.array([7.,8.,9.])
ref_hypodepth = 25.
ref_rhypo = np.linspace(5.,550,num=60)
g = 9.81

fullfile = pd.read_csv(full_file_path)

stlon = fullfile['Slon']
stlat = fullfile['Slat']

##############################################################################

########   Pull out all data   ########
 
### first, open the pandas flatfile
#thresh0data = pd.read_csv(threshold0_path)
#thresh1data = pd.read_csv(threshold1_path)
#thresh5data = pd.read_csv(threshold5_path)
#thresh10data = pd.read_csv(threshold10_path)
#
### Refine them to be within 500 km hypocentral distance, and reverse focal mechanism:
#
#refine_thresh0 = thresh0data[(thresh0data.rhypo <= 500) & (thresh0data.mechanism == 'Reverse')]
#refine_thresh1 = thresh1data[(thresh1data.rhypo <= 500) & (thresh1data.mechanism == 'Reverse')]
#refine_thresh5 = thresh5data[(thresh5data.rhypo <= 500) & (thresh5data.mechanism == 'Reverse')]
#refine_thresh10 = thresh10data[(thresh10data.rhypo <= 500) & (thresh10data.mechanism == 'Reverse')]
#
#
### Get values to set up another dict:
#eqnumber = refine_thresh0.eqnumber.values
#eventname = refine_thresh0.eventname.values
#country = refine_thresh0.country.values
#origintime = refine_thresh0.origintime.values
#hypolon = refine_thresh0.hypolon.values
#hypolat = refine_thresh0.hypolat.values
#hypodepth = refine_thresh0.hypodepth.values
#mw = refine_thresh0.mw.values
#m0 = refine_thresh0.m0.values
#mechanism = refine_thresh0.mechanism.values
#rake = np.full_like(mechanism,90)
#nostations = refine_thresh0.nostations.values
#
#network = refine_thresh0.network.values
#station = refine_thresh0.station.values
#stlon = refine_thresh0.stlon.values
#stlat = refine_thresh0.stlat.values
#stelev = refine_thresh0.stelev.values
#instrumentcode = refine_thresh0.instrumentcode.values
#rhypo = refine_thresh0.rhypo.values
#
#duration_0cm = refine_thresh0.duration_3comp.values
#duration_1cm = refine_thresh1.duration_3comp.values
#duration_5cm = refine_thresh5.duration_3comp.values
#duration_10cm = refine_thresh10.duration_3comp.values
#
#arias_0cm = refine_thresh0.arias_3comp.values
#arias_1cm = refine_thresh1.arias_3comp.values
#arias_5cm = refine_thresh5.arias_3comp.values
#arias_10cm = refine_thresh10.arias_3comp.values
#
#cav_0cm = refine_thresh0.cav_3comp.values
#cav_1cm = refine_thresh1.cav_3comp.values
#cav_5cm = refine_thresh5.cav_3comp.values
#cav_10cm = refine_thresh10.cav_3comp.values
#
#pga = refine_thresh0.pga.values
#pgv = refine_thresh0.pgv.values
#
### Get repi:
#repi = np.array([])
#for stationi in range(len(stlon)):
#    i_repi = tmf.compute_repi(stlon[stationi],stlat[stationi],hypolon[stationi],hypolat[stationi])
#    ## append to the array:
#    repi = np.append(repi,i_repi)
#
######     FIX MAGNITUDE     ######
### Ibaraki had the wrong CMT magnitude - it was listed as 7.7 but is actually 7.9.
###  The Mpgd events file has been changed, so will be reflected; but now change for the existing
### magnitde:
#ibaraki_ind = np.where(eventname == 'Ibaraki2011')
#mw[ibaraki_ind] = 7.9
#m0[ibaraki_ind] = 10**(7.9*(3/2.) + 9.1)
#
#
##############     M from PGD     ###################
### Read in the PGD csv file:
#mpgd_table = pd.read_csv(pgd_mag_file,index_col=0)
#    
### Set an empty array with the magnitude from pgd:
#mw_pgd = np.array([])
#
### Loop through every entry in the refined file, and get the pgd magnitude for 
###   that event:
#for recording_i in range(len(eventname)):
#    ## The even tname for this recording
#    i_eventname = eventname[recording_i]
#    ## The existing mw for this recording:
#    i_mw = mw[recording_i]
#    
#    ## Find where in the mw pgd table this exists:
#    i_pgdtable_info = mpgd_table[mpgd_table.EQ == i_eventname]
#    
#    ## Get the CMT mag
#    i_cmtmw = i_pgdtable_info.GCMT.values
#    i_pgdmw = i_pgdtable_info.MPGD.values
#    
#    ## Check that it is the same as the mw:
#    assert i_cmtmw == i_mw, 'The pgd table MW is not the same as the known MW'
#    mw_pgd = np.append(mw_pgd,i_pgdmw)
#
### Now convert to moment:
#m0_pgd = 10**(mw_pgd*(3/2.) + 9.1)


#############     Vs30     ###################
## Set up a call to get the Vs30 from teh global file:
## Write out the station lon lat to a tmp file:
np.savetxt('tmp',np.c_[stlon,stlat],fmt='%.8f,%.8f',delimiter=',')

## Call grdtrack with the vs30 file:
calltext="%sgmt grdtrack tmp -G%s > tmp2" %(gmt_path,vs30grdfile)

print('Calling: ' + calltext)

# Make system call
command=split(calltext)
p = subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
out,err = p.communicate()

# Print output
print(out)
print(err)
log=str(out) + str(err)

## Read in the output file:
vs30_data = np.genfromtxt('tmp2',usecols=2)

pd.DataFrame(vs30_data).to_csv('/home/eking/Documents/internship/data/Kappa/vs30.csv')

kappa_data = fullfile[' tstar(s) ']


#############    GMPEs     ###################
imt_pga = imt.PGA()
imt_pgv = imt.PGV()
imt_arias = imt.IA()
uncertaintytype = const.StdDev.TOTAL

## Set GMPEs:
zhao2006 = ZhaoEtAl2006SInter()
travasarou = TravasarouEtAl2003()
bssa14 = BooreEtAl2014()

## Set the empty arrays:
median_zhao2006 = np.array([])
median_travasarou = np.array([])
median_bssa14 = np.array([])

sd_zhao2006 = np.array([])
sd_travasarou = np.array([])
sd_bssa14 = np.array([])

## Run per recording:
for recording_i in range(len(station)):
    ## Make the rupture and distance contexts:
    i_rctx = RuptureContext()
    i_dctx = DistancesContext()
    i_sctx = SitesContext()
    
    # Zhao & Travasarou wants rrup, but dont' have that - so set to rhypo;
    # BSSA14 wants rjb but don't have it, so set to repi:
    i_dctx.rrup = rhypo[recording_i]
    i_dctx.rjb = repi[recording_i]
    
    ## Site:
    i_sctx.vs30 = np.array([vs30_data[recording_i]])
    
    ## Rupture - USE THE PGD MAGNTIUDE!
    i_rctx.rake = rake[recording_i]
    i_rctx.mag = mw_pgd[recording_i]
    i_rctx.hypo_depth = hypodepth[recording_i]
    
    
    ## Get thje Zhao predictions (in g), Travasarou (unitless), and BSSA (cm/s)
    i_median_zhao2006, i_sd_zhao2006 = zhao2006.get_mean_and_stddevs(i_sctx, i_rctx, i_dctx, imt_pga, [const.StdDev.TOTAL])
    i_median_travasarou, i_sd_travasarou = travasarou.get_mean_and_stddevs(i_sctx, i_rctx, i_dctx, imt_arias, [const.StdDev.TOTAL])
    i_median_bssa14, i_sd_bssa14 = bssa14.get_mean_and_stddevs(i_sctx, i_rctx, i_dctx, imt_pgv, [const.StdDev.TOTAL])
    
    ## Convert BSSA from cm/s to m/s, and keep in linear space:
    i_median_bssa14 = np.exp(i_median_bssa14) * 1e-2
    
    ## Convert Zhao from g to m/s/s. and keep in linear space:
    i_median_zhao2006 = np.exp(i_median_zhao2006) * g
    
    ## Put the travasarou median into linear space:
    i_median_travasarou = np.exp(i_median_travasarou)
    
    
    ## Appnd to arrays
    median_zhao2006 = np.append(median_zhao2006,i_median_zhao2006)
    median_travasarou = np.append(median_travasarou,i_median_travasarou)
    median_bssa14 = np.append(median_bssa14,i_median_bssa14)
    
    ## But keep the standard deviations in their original log units....    
    sd_zhao2006 = np.append(sd_zhao2006,i_sd_zhao2006[0])
    sd_travasarou = np.append(sd_travasarou,i_sd_travasarou[0])
    sd_bssa14 = np.append(sd_bssa14,i_sd_bssa14[0])

    
## Finally, compute the residuals:
## Compute the residuals in ln space - so get ln of the observed values divided by median:
zhao_residual = np.log(pga / median_zhao2006)
bssa_residual = np.log(pgv / median_bssa14)
travasarou_residual_0cm = np.log(arias_0cm / median_travasarou)
travasarou_residual_1cm = np.log(arias_1cm / median_travasarou)
travasarou_residual_5cm = np.log(arias_5cm / median_travasarou)
travasarou_residual_10cm = np.log(arias_10cm / median_travasarou)

## Also get reference values:
ref_M = fixed_M
## Get reference theta given reference hypo deptha nd rhypo
theta = np.arccos(ref_hypodepth/ref_rhypo)

## Repi is hypo depth times tangent of theta:
ref_repi = ref_hypodepth*np.tan(theta)

## Loop through the magnitudes and get the predicted values for fixed R and M:
## Set distance context:
dctx = DistancesContext()
sctx = SitesContext()

dctx.rrup = ref_rhypo
dctx.rjb = ref_repi

sctx.vs30 = np.full_like(ref_rhypo,760)

## Initiate arrays:
ref_zhaomedian = np.zeros((len(ref_rhypo),len(ref_M)))
ref_travasaroumedian = np.zeros((len(ref_rhypo),len(ref_M)))
ref_bssamedian = np.zeros((len(ref_rhypo),len(ref_M)))

ref_zhaosd = np.zeros((len(ref_rhypo),len(ref_M)))
ref_travasarousd = np.zeros((len(ref_rhypo),len(ref_M)))
ref_bssasd = np.zeros((len(ref_rhypo),len(ref_M)))


for i_mw in range(len(ref_M)):
    ## Set rupture context:
    i_rctx = RuptureContext()
    i_rctx.mag = ref_M[i_mw]
    i_rctx.rake= 90
    i_rctx.hypo_depth = ref_hypodepth
    
    ## Get zhao, bssa, and travasarou:
    i_ref_zhaomedian, i_ref_zhaosd = zhao2006.get_mean_and_stddevs(sctx,i_rctx,dctx,imt_pga,[const.StdDev.TOTAL])
    i_ref_bssamedian, i_ref_bssasd = bssa14.get_mean_and_stddevs(sctx,i_rctx,dctx,imt_pgv,[const.StdDev.TOTAL])
    i_ref_travasaroumedian, i_ref_travasarousd = travasarou.get_mean_and_stddevs(sctx,i_rctx,dctx,imt_arias,[const.StdDev.TOTAL])
    
    ## Convert BSSA from cm/s to m/s, keep in linear space:
    i_ref_bssamedian = np.exp(i_ref_bssamedian) * 1e-2
    
    ## Convert Zhao from g to m/s/s, keep in linear space:
    i_ref_zhaomedian = np.exp(i_ref_zhaomedian) * g
    
    ## Convert Travasarou to linear space:
    i_ref_travasaroumedian = np.exp(i_ref_travasaroumedian)
    
    ## Append these to the overall array, in linear space:
    ref_zhaomedian[:,i_mw] = i_ref_zhaomedian
    ref_travasaroumedian[:,i_mw] = i_ref_travasaroumedian
    ref_bssamedian[:,i_mw] = i_ref_bssamedian
    
    ## Append these to the overall array, keep in natural log space:
    ref_zhaosd[:,i_mw] = i_ref_zhaosd[0]
    ref_travasarousd[:,i_mw] = i_ref_travasarousd[0]
    ref_bssasd[:,i_mw] = i_ref_bssasd[0]
    
    
    

#############     Final     ###################
## Now, put all the final arrays together into a pandas dataframe. First mak a dict:
dataset_dict = {'eqnumber':eqnumber, 'eventname':eventname,'country':country, 'origintime':origintime,
                    'hypolon':hypolon, 'hypolat':hypolat, 'hypodepth':hypodepth, 'mw':mw, 'm0':m0, 
                    'mw_pgd':mw_pgd, 'm0_pgd':m0_pgd, 'mechanism':mechanism, 'rake':rake, 'nostations':nostations,  
                    'network':network, 'station':station, 'stlon':stlon, 'stlat':stlat, 'stelev':stelev,
                    'vs30':vs30_data,'instrumentcode':instrumentcode, 'rhypo':rhypo, 'repi':repi,
                    'duration_0cm':duration_0cm, 'duration_1cm':duration_1cm, 
                    'duration_5cm':duration_5cm, 'duration_10cm':duration_10cm, 
                    'arias_0cm':arias_0cm, 'arias_1cm':arias_1cm, 
                    'arias_5cm':arias_5cm, 'arias_10cm':arias_10cm, 
                    'cav_0cm':cav_0cm, 'cav_1cm':cav_1cm, 
                    'cav_5cm':cav_5cm, 'cav_10cm':cav_10cm, 'pga':pga, 'pgv':pgv,
                    'zhaopr':median_zhao2006, 'travasaroupr':median_travasarou,
                    'bssapr':median_bssa14, 'zhaosd':sd_zhao2006, 'travasarousd':sd_travasarou,
                    'bssasd':sd_bssa14, 'residual_zhao':zhao_residual, 'residual_bssa':bssa_residual,
                    'residual_trav0cm':travasarou_residual_0cm, 'residual_trav1cm':travasarou_residual_1cm,
                    'residual_trav5cm':travasarou_residual_5cm, 'residual_trav10cm':travasarou_residual_10cm}

## Put reference info into a separate dict:
reference_gmpe_dict = {'refrhypo':ref_rhypo, 'refrepi':ref_repi, 'refVs30':sctx.vs30, 
                    'refmedian_zhao_M7':ref_zhaomedian[:,0], 'refmedian_zhao_M8':ref_zhaomedian[:,1],
                    'refmedian_zhao_M9':ref_zhaomedian[:,2], 'refsd_zhao_M7':ref_zhaosd[:,0],
                    'refsd_zhao_M8':ref_zhaosd[:,1], 'refsd_zhao_M9':ref_zhaosd[:,2],
                    'refmedian_travasarou_M7':ref_travasaroumedian[:,0], 'refmedian_travasarou_M8':ref_travasaroumedian[:,1],
                    'refmedian_travasarou_M9':ref_travasaroumedian[:,2], 'refsd_travasarou_M7':ref_travasarousd[:,0],
                    'refsd_travasarou_M8':ref_travasarousd[:,1], 'refsd_travasarou_M9':ref_travasarousd[:,2],
                    'refmedian_bssa_M7':ref_bssamedian[:,0], 'refmedian_bssa_M8':ref_bssamedian[:,1],
                    'refmedian_bssa_M9':ref_bssamedian[:,2], 'refsd_bssa_M7':ref_bssasd[:,0],
                    'refsd_bssa_M8':ref_bssasd[:,1], 'refsd_bssa_M9':ref_bssasd[:,2]}

## Put these all into a dataframe:
dataset_df = pd.DataFrame(data=dataset_dict)
reference_gmpe_df = pd.DataFrame(reference_gmpe_dict)

## SAve to a file for analysis:
dataset_df.to_csv(refinedflatfile_path,index=False)
reference_gmpe_df.to_csv(refgmpe_path,index=False)




