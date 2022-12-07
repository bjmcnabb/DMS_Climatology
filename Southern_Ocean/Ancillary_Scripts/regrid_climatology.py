# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 10:52:12 2022

@author: bcamc
"""

def regrid_climatology(climatology, DMS, grid, bounds, to_mask, var_months_, interp_method='griddata'):
    import pandas as pd
    import numpy as np
    
    from bin2D import bin2D
    from interp2D import interp2D
    
    # interp_method = 'RBF' or 'griddata'
    idx = pd.IndexSlice
    
    #-----------------------------------------------------------------
    #### Bin data
    print('binning climatology...')
    data_gridded = bin2D(climatology, grid, bounds)
    
    
    #### Create land/ice mask
    
    # interpolate main data
    print('interpolating & reindexing climatology...')
    data_gridded = data_gridded.squeeze()
    data_interp = interp2D(data_gridded, interp_method=interp_method)

    # Reindex data to match land mask
    if to_mask.index.levels[2][0]-data_interp.index.levels[2][0] != 0 or to_mask.levels[1][0]-data_interp.index.levels[1][0] != 0:
        # extract difference in idx
        lon_corr = to_mask.index.levels[2][0]-data_interp.index.levels[2][0]
        lat_corr = to_mask.index.levels[1][0]-data_interp.index.levels[1][0]
        # reset idx and apply correction
        data_interp = data_interp.reset_index()
        data_interp['latbins'] = data_interp['latbins']+lat_corr
        data_interp['lonbins'] = data_interp['lonbins']+lon_corr
        # round the idx for an exact match
        data_interp['latbins'] = data_interp['latbins'].round(3)
        data_interp['lonbins'] = data_interp['lonbins'].round(3)
        # convert back to indexed series 
        data_interp = data_interp.set_index(['datetime','latbins','lonbins']).squeeze('columns')
        # finally, reindex
        data_interp_reind = data_interp.reindex_like(to_mask)
    
    # reinterpolate - fills in gaps produced from the reindexing
    print('reinterpolate climatology to fill in reindexing gaps...')
    data_interp_reind = data_interp_reind.squeeze()
    land_ice_mask = interp2D(data_interp_reind, interp_method=interp_method)
    
    #### Create unrestricted interpolation
    
    # key difference - this removes the original masking so the interpolation function creates a larger convex hull
    data_gridded[data_gridded == 0] = np.nan 
    
    # interpolate main data
    print('reinterpolate climatology as a land/ice mask...')
    data_gridded = data_gridded.squeeze()
    data_interp = interp2D(data_gridded, interp_method=interp_method)
    
    # Reindex data to match land mask
    print('reindex climatology as a land/ice mask...')
    if to_mask.index.levels[2][0]-data_interp.index.levels[2][0] != 0 or to_mask.levels[1][0]-data_interp.index.levels[1][0] != 0:
        # extract difference in idx
        lon_corr = to_mask.index.levels[2][0]-data_interp.index.levels[2][0]
        lat_corr = to_mask.index.levels[1][0]-data_interp.index.levels[1][0]
        # reset idx and apply correction
        data_interp = data_interp.reset_index()
        data_interp['latbins'] = data_interp['latbins']+lat_corr
        data_interp['lonbins'] = data_interp['lonbins']+lon_corr
        # round the idx for an exact match
        data_interp['latbins'] = data_interp['latbins'].round(3)
        data_interp['lonbins'] = data_interp['lonbins'].round(3)
        # convert back to indexed series 
        data_interp = data_interp.set_index(['datetime','latbins','lonbins']).squeeze('columns')
        # finally, reindex
        data_interp_reind = data_interp.reindex_like(to_mask)
    
    # reinterpolate - fills in gaps produced from the reindexing
    print('interpolate land/ice mask and mask climatology...')
    data_interp_reind = data_interp_reind.squeeze()
    data_unmasked = interp2D(data_interp_reind, interp_method=interp_method)
    
    ### Index out land/ice
    data_final = data_unmasked.where(land_ice_mask!=0,np.nan)
    data_final = data_final.where(to_mask.notna(),np.nan)
    
    #### Final correction where mask fails
    latmask = [-80, -74]
    lonmask = [-110, -70]
    data_final.loc[idx[:,latmask[0]:latmask[1], lonmask[0]:lonmask[1]]] = np.nan
    
    return data_final
