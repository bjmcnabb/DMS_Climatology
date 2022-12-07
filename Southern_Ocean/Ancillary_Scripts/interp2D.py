# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 16:28:38 2022

@author: bcamc
"""

def interp2D(data_input, interp_method='griddata'):
    import numpy as np
    import pandas as pd
    import scipy
    from tqdm import tqdm
    # interpolate main data
    data_interp = {}
    for j, month in tqdm(enumerate(data_input.index.get_level_values('datetime').unique().astype('int'))):
        if interp_method == 'griddata':
            # create 2 collumn array with lat/lon coordinates to interpolate
            coords = np.stack([data_input.loc[month,:].index.get_level_values('lonbins').values,
                               data_input.loc[month,:].index.get_level_values('latbins').values],axis=1)
            
            # index actual data and coordinates (i.e. remove nans)
            # ind = data_input.loc[month,:].notna() # filter out nans first
            lon_pts = data_input.loc[month,:].dropna().index.get_level_values('lonbins').values
            lat_pts = data_input.loc[month,:].dropna().index.get_level_values('latbins').values
            values = data_input.loc[month,:].dropna().values
            
            # interpolate data using a convex hull and linear function
            interpd = scipy.interpolate.griddata(points=np.stack([lon_pts,lat_pts],axis=1),
                                                  values=values,
                                                  xi=coords,
                                                  method='linear')
            
            # create a series of the interpolated data
            interpd = pd.Series(data=interpd, index=data_input.loc[month,:].index)
            # Restrict interpolation to original data min/max bounds
            interpd.loc[interpd>np.nanmax(data_input.loc[month,:])] = np.nanmax(data_input.loc[month,:])
            interpd.loc[interpd<np.nanmin(data_input.loc[month,:])] = np.nanmin(data_input.loc[month,:])
            # append each month together
            data_interp[float(month)] = interpd
            
        elif interp_method == 'RBF':
            # create 2 collumn array with lat/lon coordinates to interpolate
            coords = np.stack([data_input.loc[[month],:].index.get_level_values('datetime').values,
                                data_input.loc[month,:].index.get_level_values('latbins').values,
                                data_input.loc[month,:].index.get_level_values('lonbins').values],axis=1)
            
            # index actual data and coordinates (i.e. remove nans)
            date_pts = data_input.loc[[month],:].dropna().index.get_level_values('datetime').values
            lon_pts = data_input.loc[month,:].dropna().index.get_level_values('lonbins').values
            lat_pts = data_input.loc[month,:].dropna().index.get_level_values('latbins').values
            values = data_input.loc[month,:].dropna().values
            
            chunk_size = 10000
            
            # interpolate
            interpd = scipy.interpolate.RBFInterpolator(da.from_array(np.stack([date_pts,lat_pts,lon_pts],axis=1),chunks=chunk_size),
                                                        da.from_array(values, chunks=chunk_size),
                                                        kernel='gaussian',
                                                        epsilon=2,
                                                        neighbors=50)(da.from_array(coords,chunks=chunk_size))
                                
            # create a series of the interpolated data
            interpd = pd.Series(data=interpd, index=data_input.loc[month,:].index)
            # Restrict interpolation to original data min/max bounds
            interpd.loc[interpd>np.nanmax(data_input.loc[month,:])] = np.nanmax(data_input.loc[month,:])
            interpd.loc[interpd<np.nanmin(data_input.loc[month,:])] = np.nanmin(data_input.loc[month,:])
            # save our appended list as a single series inside the dict, and add dates back in
            data_interp[float(month)] = interpd
        
    # concatenate files
    data_interp = pd.concat(data_interp, names=['datetime','latbins','lonbins'])
    return data_interp
    