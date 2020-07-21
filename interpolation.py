#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 13:58:54 2019

@author: eking
"""
import numpy as np
import pandas as pd
import scipy
from matplotlib import pyplot as plt
## Another way to do this is to make a meshed grid and plot it with different python functions (contour,pcolor,etc.)
## To do this we'll need to loop through the unique longitudes and latitudes to make a "grid" of SST (draw on board)

## the end goal of what we want are three 2d arrays. That way, we can hand off all three arrays to python gridding fn's.
## We will need three things:
## 1. A grid of X points:
## [x1,x2,x3,x4,x5]
## [x1,x2,x3,x4,x5]
## ...
## 2. A grid of the Y points:
## [y1,y1,y1,y1,y1]
## [y2,y2,y2,y2,y2]
## ...
## 3. A grid of the Z points that correspond to the X and Y grids:
## [Z_x1y1, Z_x2y1, Z_x3y1, Z_x4y1, Z_x5y1]
## [Z_x1y2, Z_x2y2, Z_x3y2, Z_x4y2, Z_x5y2]
## ...
## That way each point combination (index) between thtese three arrays can be used together to match for a grid.
#%%
import numpy as np
from scipy import interpolate

data = pd.read_csv('/home/eking/Documents/internship/data/Kappa/SNR_3/75_bins/15/full_file.csv')

lon = data['Slon']
lat = data['Slat']
kappa = data[' tstar(s) ']

minlat = 36
maxlat = 39

minlon = -124
maxlon = -120.5


X = np.linspace(minlon,maxlon,4)
Y = np.linspace(minlat,maxlat,4)

x,y = np.meshgrid(X,Y)


f = interpolate.interp2d(lon,lat,kappa,kind='cubic')

# use linspace so your new range also goes from 0 to 3, with 8 intervals
Xnew = np.linspace(minlon,maxlat,8)
Ynew = np.linspace(minlat,maxlat,8)

test8x8 = f(Xnew,Ynew)

plt.figure()
plt.contour(GRIDLON,,kappa_grid,40,linewidths=0.8,cmap='magma')

## Add a colorbar. Don't need any input if you don't want to, so it's empty parentheses.
cb=plt.colorbar()
## Set a label on the colorbar
cb.set_label('SST (deg. C)')
plt.show()

#%%
data = pd.read_csv('/home/eking/Documents/internship/data/Kappa/SNR_3/75_bins/15/full_file.csv')

lon = data['Slon']
lat = data['Slat']
kappa = data[' tstar(s) ']

minlat = 36
maxlat = 39

minlon = -124
maxlon = -120.5
print(kappa)

numberofpoints = 30

from scipy import interpolate
f = interpolate.interp2d(lon,lat,kappa,kind='cubic')

gridlon = np.linspace(minlon,maxlon,numberofpoints)
gridlat = np.linspace(minlat,maxlat,numberofpoints)


GRIDLON,GRIDLAT = np.meshgrid(gridlon,gridlat)

lonshp = np.shape(GRIDLON)
latshp = np.shape(GRIDLAT)

GRIDLON = np.reshape(GRIDLON, -1)
GRIDLAT = np.reshape(GRIDLAT, -1)


kappa_grid = f(GRIDLON,GRIDLAT)

GRIDLON = np.reshape(GRIDLON, lonshp)
GRIDLAT = np.reshape(GRIDLAT, latshp)

print(GRIDLON)
print(GRIDLAT)
print(kappa_grid)

#%%
plt.figure()
plt.contour(GRIDLON,GRIDLAT,kappa_grid,40,linewidths=0.8,cmap='magma')

## Add a colorbar. Don't need any input if you don't want to, so it's empty parentheses.
cb=plt.colorbar()
## Set a label on the colorbar
cb.set_label('SST (deg. C)')
plt.show()
