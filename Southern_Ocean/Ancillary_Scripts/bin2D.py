# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 16:25:35 2022

@author: bcamc
"""

def bin2D(input_data, grid, bounds):
    import numpy as np
    import pandas as pd
    from tqdm import tqdm
    min_lat, max_lat, min_lon, max_lon = bounds
    
    for i,j in tqdm(enumerate(input_data.index.get_level_values('datetime').unique())):
        data = input_data.loc[[j],:].unstack('lonbins').values
        lat = input_data.loc[[j],:].unstack('lonbins').index.get_level_values('latbins').values
        lon = input_data.loc[[j],:].unstack('lonbins').columns.values
        time = input_data.loc[[j],:].unstack('lonbins').index.get_level_values('datetime').unique().values
        # case 1: match pixels to high resolution satellite data
        if lon[1]-lon[0] >= grid: #i.e. if upsampling to finer grid size
            # Regrid data and interpolate though:
            # First find the indices to index by lat/lon
            latinds = np.argwhere((lat >= min_lat) & (lat <= max_lat)).astype(int)
            loninds = np.argwhere((lon >= min_lon) & (lon <= max_lon)).astype(int)
        
            # Restrict the data to the specified lat/lons
            data_indexed = data[latinds[:,None], loninds[None,:]][:,:,0]
        
            # Create data matrix with coordinates
            data_mat = pd.DataFrame(data_indexed)
            data_mat.columns = lon[loninds].flatten()
            data_mat.index = lat[latinds].flatten()
            # sort the data so that columns are ascending (matches 'new_shape' below)
            data_mat = data_mat.reindex(sorted(data_mat.columns), axis=1) # sort columns in ascending order
        
            # Create a new matrix with new gridded coordinates
            lat_new = np.arange(min_lat, max_lat+grid, grid).round(3)
            lon_new = np.arange(min_lon, max_lon+grid, grid).round(3)
            new_shape = pd.DataFrame(np.ones((lat_new.shape[0],lon_new.shape[0])))
            new_shape.columns=lon_new
            new_shape.index=lat_new
        
            # Now reindex data to new coordinates - add in NaNs to interpolate through
            # important: find corresponding nearest value in new index/columns to replace data coords by
            new_idx = [new_shape.index[abs(new_shape.index.values-i).argmin()] for i in data_mat.index]
            new_cols = [new_shape.columns[abs(new_shape.columns.values-i).argmin()] for i in data_mat.columns]
            # now rename idx/cols with new values in the data matrix (this allows the reindex_like function to properly map and insert nans below)
            data_mat.set_axis(new_idx, axis=0, inplace=True)
            data_mat.set_axis(new_cols, axis=1, inplace=True)
        
            data_proc = pd.DataFrame(data_mat.reindex_like(new_shape).stack(dropna=False))
            data_proc = data_proc.rename(columns={0:'data'})
            data_proc.index = data_proc.index.set_names(['latbins','lonbins'])
            data_proc['datetime'] = np.tile(time,data_proc.shape[0])
            data_proc.reset_index(inplace=True)
            data_proc.set_index(['datetime','latbins','lonbins'], inplace=True)
            # data_proc.reset_index(inplace=True)
            
        #-----------------------------------------------------------------
        # case 2: downsample to courser grid
        if lon[1]-lon[0] < grid: # i.e. if interpolating to courser grid
            # Bin the data
            
            # First find the indices to index by lat/lon
            latinds = np.argwhere((lat >= min_lat) & (lat <= max_lat)).astype(int)
            loninds = np.argwhere((lon >= min_lon) & (lon <= max_lon)).astype(int)
            
            # Restrict the data to the NE Pacific lat/lons
            data_indexed = data[latinds[:,None], loninds[None,:]][:,:,0]
            
            # Convert time/lats/lons into repeating matrices to match data dimensions
            lat_mat = np.tile(lat[latinds], len(lon[loninds])) # latitude is repeated by column
            lon_mat = np.tile(lon[loninds], len(lat[latinds])).T # transpose to repeat longitude by row
            time_mat = np.tile(time, len(np.ravel(lat_mat)))
            
            # Now create a new matrix with the unravel matrices into vectors - np.ravel does this row-by-row
            # Need these as pandas dataframes to using binning scheme:
            d = {'datetime':time_mat,'lat':np.ravel(lat_mat),'lon':np.ravel(lon_mat),'data':np.ravel(data_indexed)}
            data_long = pd.DataFrame(data=d)
            
            # Bin data as averages across gridded spatial bins:
            to_bin = lambda x: np.round(x / grid) * grid
            data_long['latbins'] = data_long.lat.map(to_bin)
            data_long['lonbins'] = data_long.lon.map(to_bin)
            data_proc = data_long.groupby(['datetime', 'latbins', 'lonbins']).mean()
            
            # Rename binned columns + drop mean lat/lons:
            data_proc = data_proc.drop(columns=['lat', 'lon'])
            # data_proc.reset_index(inplace=True) # remove index specification on columns
        if i==0:
            data_binned = data_proc
        else:
            data_binned = pd.concat([data_binned, data_proc],axis=0)
    return data_binned