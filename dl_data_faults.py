## Download events and stations for proposal
# VJS 5/2018

from obspy.clients.fdsn.client import Client
from obspy.core.utcdatetime import UTCDateTime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import cartopy.feature as cfeature
import matplotlib.ticker as mticker


## Paths and Shit ##
working_dir = '/home/eking/Documents/internship/data/Kappa/'
#eventfile = '/Users/vjs/ridgecrest2019/data/events/ridgecrest_events_jul9.txt'
gmt_latestquat = working_dir + 'kappa_maps/latest_quaternary.txt'
ascii_latestquat = working_dir + 'kappa_maps/latest_quaternary_BIGzoomed_bayA.pckl'
#ge_eventfile = '/Users/vjs/ridgecrest2019/data/events/ridgecrest_events_forge.txt'
#deployment_dir = '/Users/vjs/ridgecrest2019/deployment_pattern/'

minlat = 37.75
maxlat = 37.95
minlon = -122.35
maxlon = -122.15

#-122.35, -122.15, 37.75, 37.95
#-122.5  -121.5  37.5 38.5
# -124   -120.5  36  39
#starttime = UTCDateTime("2019-07-04")
#endtime = UTCDateTime("2019-07-10")


##########  Define conversion for usgs faults    #########
def multiseg2pckl_py3(multisegpath,pcklpath,pathlimits):
    '''
    VJS 7/2019
    Convert a GMT multisegment file to a pckl file to be plotted in python
    Input:
        multisegpath:       String with the path to the input multisegment file
        pcklpath:           String with the path to the output pckl file; List 
                            of arrays, each with a segment to scatter or plot
                            Output to pcklpath.
        pathlimits:         [[lonmin,lonmax],[latmin,latmax], same as
                            [[xmin,xmax],[ymin,ymax]] 
    Output:
        allsegments         List of arrays, each with two columns (lon, lat)
    '''
    
    from numpy import zeros,array,where
    import matplotlib.path as mplPath
    import pickle as pickle
    
    #Get the corner coordinates out of pathlimits: first lonmin/max, latmin/max:
    lonmin=pathlimits[0][0]
    lonmax=pathlimits[0][1]
    latmin=pathlimits[1][0]
    latmax=pathlimits[1][1]
    
    #Now the actual corners:
    bottom_left=[lonmin,latmin]
    bottom_right=[lonmax,latmin]
    top_right=[lonmax,latmax]
    top_left=[lonmin,latmax]
    
    #Define the path bounds - for a squre, it's 5 points:
    path_coordinates=array([bottom_left,bottom_right,top_right,top_left,bottom_left])
    #Define the regionpath with the mplPath command:
    region_path=mplPath.Path(path_coordinates)
    
    #First count the number of segments and the number of elements in each segment
    Nsegments=0
    Num_elements=[]
    f=open(multisegpath,'r')
    first_line=True
    while True:
        line=f.readline()
        if '>' in line:
            if first_line==False: #Append previous count
                Num_elements.append(Numel)
            first_line=False
            Nsegments+=1
            Numel=0
        else:
            Numel+=1
        if line=='': #End of file
            Num_elements.append(Numel-1)
            break
    f.close()
            
    #Now loop over segments and make an arra per segment adn append to list of arrays
    all_segments=[]    
    f=open(multisegpath,'r')
    
    for ksegment in range(Nsegments):
        
        #First line in the segmetn is the stupid carrot
        line=f.readline()
        
        #Now read the next Num_elements[ksegment] lines
        lonlat=zeros((Num_elements[ksegment],2))
        for kelement in range(Num_elements[ksegment]):
            line=f.readline()
            lonlat[kelement,0]=float(line.split()[0])
            lonlat[kelement,1]=float(line.split()[1])
            
        
        #Before appending this segment to the list, check if any points along 
        #the segment are in the path
        
        #Are any points of this segment in the path defined above?
        points_logical=where(region_path.contains_points(lonlat)==True)[0]
        
        #If any points along htis segment are contained:
        if len(points_logical>0):  
            #Done, append to list
            all_segments.append(lonlat)
    
    f.close()
    
    #Write to the pickle file:
    fout=open(pcklpath,'wb')
    for segment_i in range(len(all_segments)):
        pickle.dump(all_segments[segment_i],fout)
    fout.close()
    
    return all_segments

######################################################



### Convert the faults file to a multisegment file to plot:
quatfaults = multiseg2pckl_py3(gmt_latestquat,ascii_latestquat,[[minlon,maxlon],[minlat,maxlat]])
#
### Opoen deployment data:
#grid1 = np.genfromtxt(deployment_dir + 'grid1.txt')
#grid2 = np.genfromtxt(deployment_dir + 'grid2.txt')
#grid3 = np.genfromtxt(deployment_dir + 'grid3.txt')
#grid4 = np.genfromtxt(deployment_dir + 'grid4.txt')
#
#evClient = Client("USGS")
#ev_inventory = evClient.get_events(starttime=starttime, endtime=endtime, minlatitude=minlat, maxlatitude=maxlat, minlongitude=minlon, maxlongitude=maxlon, minmagnitude=1.0, maxmagnitude=None)
#
#ev_lon = []
#ev_lat = []
#ev_mag = []
#ev_utcdt = []
#ev_day = []
#
#for i_event in range(len(ev_inventory)):
#    eventi = ev_inventory[i_event]
#    ev_mag.append(eventi.magnitudes[0]['mag'])
#    ev_lon.append(eventi.origins[0]['longitude'])
#    ev_lat.append(eventi.origins[0]['latitude'])
#    ev_utcdt.append(eventi.origins[0]['time'])
#    ev_day.append(eventi.origins[0]['time'].day)
#    
### Turn into array:
#ev_mag = np.array(ev_mag)
#ev_lon = np.array(ev_lon)
#ev_lat = np.array(ev_lat)
#ev_mag = np.array(ev_mag)
#ev_utcdt = np.array(ev_utcdt)
#ev_day = np.array(ev_day)
#    
#
### Get m7.1 and m6.4 indices:
#ind71 = np.where(ev_mag > 7.0)[0]
#ind64 = np.where((ev_mag > 6.3) & (ev_mag < 7.0))[0]
#
#
#
## Create a figure
#fig = plt.figure(figsize=(10,10))
## Create a GeoAxes 
#ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
#
#
## Limit the extent of the map to a small longitude/latitude range.
#ax.set_extent([minlon, maxlon, minlat, maxlat])
##ax.stock_img()
##ax.coastlines()
#
### Scatter events:
#eventplot = ax.scatter(ev_lon,ev_lat,c=ev_day,edgecolor='k',alpha=0.4,s=7*(np.log(10**ev_mag)),cmap='YlOrRd')
### Plot large ones on top"
#ax.scatter(ev_lon[ind64],ev_lat[ind64],marker='*',facecolor='yellow',edgecolor='k',s=40*(np.log(10**ev_mag[ind64])),label='M6.4')
#ax.scatter(ev_lon[ind71],ev_lat[ind71],marker='*',facecolor='orange',edgecolor='k',s=40*(np.log(10**ev_mag[ind71])),label='M7.1')
#
### Colorbar
#cbar = plt.colorbar(eventplot)
#cbar.set_label('UTC Day in July')
#cbar.set_alpha(1)
#cbar.draw_all()
#
### Plot coso:
#ax.scatter(-117.8325,36.1708,marker='P',s=200,facecolor='green',edgecolor='black',label='Coso Geothermal Field')
#
### Plot faults:
#for i_fault in range(len(quatfaults)):
#    i_faultseg = quatfaults[i_fault]
#    ax.plot(i_faultseg[:,0],i_faultseg[:,1],c='k',linewidth=0.8)
#    
### Add deployment patterns:
#ax.scatter(grid1[:,0],grid1[:,1],marker='^',facecolor='green',edgecolor='black',label='Planned nodes')
#ax.scatter(grid2[:,0],grid2[:,1],marker='^',facecolor='green',edgecolor='black')
#ax.scatter(grid3[:,0],grid3[:,1],marker='^',facecolor='green',edgecolor='black')
#ax.scatter(grid4[:,0],grid4[:,1],marker='^',facecolor='green',edgecolor='black')
#    
### Legend:
#ax.legend(loc=4)
#
### Add gridlines:
#gl_minor = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,linewidth=1, color='gray', alpha=0.5, linestyle='--')
#gl_minor.xlocator = mticker.FixedLocator([-118.25,-118,-117.75,-117.5,-117.25,-117])
#gl_minor.ylocator = mticker.FixedLocator([35,35.25,35.5,35.75,36,36.25])
#
#gl_major = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,linewidth=1, color='gray', alpha=0.5, linestyle='-')
#gl_major.xlocator = mticker.FixedLocator([-118.5,-118,-117.5,-117])
#gl_major.ylocator = mticker.FixedLocator([35,35.5,36,36.5])
#
#
#
#plt.setp(ax.get_xticklabels(), fontsize=14)
#plt.setp(ax.get_yticklabels(), fontsize=14)
#
#
#plt.savefig(working_dir + 'kappa_maps/faults/faultmap.pdf')
#
#
### Save events to file for google earth:
##np.savetxt(ge_eventfile,np.c_[ev_lon,ev_lat,ev_mag,ev_day],fmt='%.8f\t%.8f\t%.1f\t%i')
#
#
#
