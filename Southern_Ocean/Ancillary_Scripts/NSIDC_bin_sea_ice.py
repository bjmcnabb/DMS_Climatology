# -*- coding: utf-8 -*-
"""
Created on Fri May  5 11:43:40 2023

@author: bcamc
"""

#%% Binning NSIDC ice concentrations

# This code takes the NSIDC sea ice concentration datasets and converts them
# from a 25 km polar sterographic projection (irregularly-spaced grid) to a 20 km 
# geodesic (regularly-spaced grid) for interpolation later.

import pyproj
import os
import xarray as xr
import pandas as pd
import numpy as np
from obspy.geodetics import kilometers2degrees

directory = 'E:/Sea_Ice/select_monthly' # E: or C:, depending if using hard drive/thumb drive
files = os.listdir(directory)
for i, file in enumerate(files):
    print('iter: '+str(i+1)+'/'+str(len(files)))
    var_ = xr.open_dataset(directory+'/'+file)
    ice_conc = var_.cdr_seaice_conc_monthly.values[0,:,:] # concentrations
    x = var_.xgrid.values  # x coordinate in km
    y = var_.ygrid.values  # y coordinate in km
    time = int(pd.to_datetime(var_.time.values).strftime('%m').values)

    # reproject x/y coordinates into the lat/lon space
    proj = pyproj.Transformer.from_crs(3412, 4326, always_xy=True)
    X,Y = np.meshgrid(x,y)
    ilon, ilat = proj.transform(X, Y)
    time_mat = np.tile(time, len(np.ravel(ilat)))

    # create indexed dataframe
    d = {'datetime':time_mat,'lat':ilat.ravel(),'lon':ilon.ravel(),'ice':ice_conc.ravel()}
    data = pd.DataFrame(d)
    
    # Bin data
    grid = kilometers2degrees(20)
    to_bin = lambda x: np.round(x / grid) * grid
    data['latbins'] = data.lat.map(to_bin)
    data['lonbins'] = data.lon.map(to_bin)
    data_proc = data.groupby(['datetime', 'latbins', 'lonbins']).mean()
    data_proc = data_proc.drop(['lat','lon'], axis=1)
    data_proc = data_proc.squeeze()
    
    # save data:
    if i == 0:
        ice = data_proc
    else:
        ice = pd.concat([ice,data_proc]).groupby(['datetime','latbins','lonbins']).mean()