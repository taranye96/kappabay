# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 15:46:19 2019

@author: Elias
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
import datetime
import pytz
import dateutil.parser
import seaborn as sns; sns.set(style="white", color_codes=True)

#import cmocean

stationfile = '/home/eking/Documents/internship/data/kappa_stations2000-2018.csv'
eventfile = '/home/eking/Documents/internship/data/kappa_events2000-2018.csv'

quakes = pd.read_csv(eventfile)

lat = quakes['latitude']
lon = quakes['longitude']
M = quakes['magnitude']
depth = quakes['depth']
time = quakes['time']

#plt.scatter(lon,lat)
#%%
# Create a Stamen Terrain instance.
stamen_terrain = cimgt.Stamen('terrain-background')

fig = plt.figure(figsize=(10,10))


# Create a GeoAxes in the tile's projection.
ax = fig.add_subplot(1, 1, 1, projection=stamen_terrain.crs)

# Limit the extent of the map to a small longitude/latitude range.
ax.set_extent([-124, -120.5, 36, 39], crs=ccrs.Geodetic())

# Add the Stamen data at zoom level 8.
ax.add_image(stamen_terrain, 8)

scat=ax.scatter(lon,lat,s=10*np.exp(M/1.5),c=depth,cmap='magma_r', alpha=0.7, transform=ccrs.PlateCarree())



mapgl = ax.gridlines(draw_labels=True)

## Change the gridline labeling: https://scitools.org.uk/cartopy/docs/latest/matplotlib/gridliner.html
mapgl.xlabels_top=False
mapgl.ylabels_left=False
mapgl.xlabel_style = {'size': 10, 'color': 'black'}
mapgl.ylabel_style = {'size': 10, 'color': 'black'}

cb = plt.colorbar(scat)

plt.savefig('/home/eking/Documents/internship/data/Quake_Map2010-2018.pdf')


#%%

stations = pd.read_csv(stationfile)

Lat = stations['latitude']
Lon = stations['longitude']
Net = stations['network']
Name = stations['name']
Elev = stations['elevation']
Chan = stations['channel']
#%%
stamen_terrain = cimgt.Stamen('terrain-background')

fig = plt.figure(figsize=(10,10))


# Create a GeoAxes in the tile's projection.
ax = fig.add_subplot(1, 1, 1, projection=stamen_terrain.crs)

# Limit the extent of the map to a small longitude/latitude range.
ax.set_extent([-124, -120.5, 36, 39], crs=ccrs.Geodetic())

# Add the Stamen data at zoom level 8.
ax.add_image(stamen_terrain, 8)

Scat=ax.scatter(Lon,Lat,s=100,c=Elev,marker='s', cmap=None, alpha=0.7, transform=ccrs.PlateCarree())



mapgl = ax.gridlines(draw_labels=True)

## Change the gridline labeling: https://scitools.org.uk/cartopy/docs/latest/matplotlib/gridliner.html
mapgl.xlabels_top=False
mapgl.ylabels_left=False
mapgl.xlabel_style = {'size': 10, 'color': 'black'}
mapgl.ylabel_style = {'size': 10, 'color': 'black'}

cb = plt.colorbar(scat)

plt.savefig('/home/eking/Documents/internship/data/Station_Map2000-2018.pdf')

#%%

stamen_terrain = cimgt.Stamen('terrain-background')

fig = plt.figure(figsize=(10,10))


# Create a GeoAxes in the tile's projection.
ax = fig.add_subplot(1, 1, 1, projection=stamen_terrain.crs)

# Limit the extent of the map to a small longitude/latitude range.
ax.set_extent([-124, -120.5, 36, 39], crs=ccrs.Geodetic())

# Add the Stamen data at zoom level 8.
ax.add_image(stamen_terrain, 8)

scat=ax.scatter(lon,lat,s=10*np.exp(M/1.5),c=depth,cmap='magma_r', alpha=0.7, transform=ccrs.PlateCarree())

Scat=ax.scatter(Lon,Lat,s=50,c=Elev,marker='s', cmap='Blues', alpha=0.7, transform=ccrs.PlateCarree())

mapgl = ax.gridlines(draw_labels=True)

## Change the gridline labeling: https://scitools.org.uk/cartopy/docs/latest/matplotlib/gridliner.html
mapgl.xlabels_top=False
mapgl.ylabels_left=False
mapgl.xlabel_style = {'size': 10, 'color': 'black'}
mapgl.ylabel_style = {'size': 10, 'color': 'black'}

cb = plt.colorbar(scat)

Cb = plt.colorbar(Scat)

plt.savefig('/home/eking/Documents/internship/data/Quake&Station_Map2000-2018.pdf')
#%%




#%%
#