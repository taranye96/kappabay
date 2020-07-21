#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 13:29:57 2019

@author: eking
"""


from obspy.clients.fdsn import Client
from obspy.core import Stream, read, UTCDateTime
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from glob import glob
import numpy as np
import csv
import pandas as pd
#%%

allmetadata = pd.read_csv('/home/eking/Documents/internship/data/flatfile.csv')

amd1,amd2,amd3,amd4,amd5 = np.array_split(allmetadata,5)

amd1.to_csv('/home/eking/Documents/internship/data/flatfile1.csv')

amd2.to_csv('/home/eking/Documents/internship/data/flatfile2.csv')

amd3.to_csv('/home/eking/Documents/internship/data/flatfile3.csv')

amd4.to_csv('/home/eking/Documents/internship/data/flatfile4.csv')

amd5.to_csv('/home/eking/Documents/internship/data/flatfile5.csv')
