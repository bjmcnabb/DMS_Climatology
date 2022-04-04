# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 09:28:55 2021

@author: bcamc
"""
#%% Reference

# CCMP Wind Speeds (<2007)
# https://podaac.jpl.nasa.gov/dataset/CCMP_MEASURES_ATLAS_L4_OW_L3_5A_MONTHLY_WIND_VECTORS_FLK

# NSIDC sea ice climatology
# https://nsidc.org/data/G02202/versions/4

# fronts
# https://www.seanoe.org/data/00486/59800/

#%% Start timer
import timeit
analysis_start = timeit.default_timer()
#%% Import Packages
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy
from scipy.stats.stats import pearsonr, spearmanr
import cartopy
import cartopy.crs as ccrs
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
import xarray as xr
from sklearn.decomposition import PCA
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn import metrics
from datetime import datetime
import datetime
import seaborn as sns
import dask.array as da
import dask.dataframe as dd
import shapely.vectorized
import cartopy.io.shapereader as shpreader
import fiona
import shapely
import shapely.geometry as sgeom
from shapely.ops import unary_union
from shapely.prepared import prep
import joblib
from obspy.geodetics import kilometers2degrees, degrees2kilometers
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import torchensemble
from torchensemble.utils.logging import set_logger
from torchensemble.utils import io
import cmocean

# Progress bar package
from tqdm import tqdm

# Import pre-built mapping functions
from SO_mapping_templates import South_1ax_map, South_1ax_flat_map
# import the custom-built neural network base model
from NN_model_frameworks import ANNRegressor
# Import taylor diagram script
from taylorDiagram import TaylorDiagram

#%% Switch Directories
dir_ = 'C:\\Users\\bcamc\\OneDrive\\Desktop\\Python\\Projects\\sulfur\\southern_ocean\\Scripts'
if os.getcwd() != dir_:
    os.chdir(dir_)
    
#%% Define Region & File Paths

#### Spatial grid resolution (degrees):
grid = kilometers2degrees(20)

#### call file directory
write_dir = 'C:/Users/bcamc/OneDrive/Desktop/Python/Projects/sulfur/southern_ocean/SO_DMS_data_v2'

#### Define lat/lon constraints
min_lon, max_lon, min_lat, max_lat = -180, 180, -90, -40

#### Define destination to save figures
save_to_path = 'C:/Users/bcamc/OneDrive/Desktop/Python/Projects/sulfur/southern_ocean/Figures'

#### Define a destination to load/save ensemble ANN models to
ANN_save_dir = 'C:/Users/bcamc/OneDrive/Desktop/Python/Projects/sulfur/southern_ocean/Scripts/'

#### For iron stress-testing, set a new directory to save models to
dir_new = 'C:\\Users\\bcamc\\OneDrive\\Desktop\\Python\\Projects\\sulfur\\southern_ocean\\Scripts\\ANN_with_FLH_chl'

#### Define bins
latbins = np.arange(min_lat,max_lat+grid,grid)
lonbins = np.arange(min_lon,max_lon+grid,grid)

# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
#### Select whether this is the first run
# NOTE: If true, tells script to interpolate data, build 1000 models ensembles,
# and save output to file (v. time consuming!). If False, load up previous output.
first_run = False
# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 

#%% Load PMEL data
# All DMS data from the PMEL repository for summertime months
PMEL = pd.read_csv('C:/Users/bcamc/OneDrive/Desktop/Python/Projects/sulfur/southern_ocean/SO_DMS_data.csv')
# filter out garbage data
PMEL = PMEL.replace(-999,np.nan)
#-----------------------------------------------------------------------------
# Extract variables
PMEL_lat = PMEL['Lat']
PMEL_lon = PMEL['Lon']
#-----------------------------------------------------------------------------
# Print metadata
print()
print('Coordinates for PMEL data:')
print('oW: ' + str([PMEL_lon.min(), PMEL_lon.max(), PMEL_lat.min(), PMEL_lat.max()]))
print('oE: ' + str([360+PMEL_lon.min(), 360+PMEL_lon.max(), PMEL_lat.min(), PMEL_lat.max()]))
print()

#%% Clean-up PMEL data
#-----------------------------------------------------------------------------

# Remove NaNs
# data_proc = PMEL.loc[:,['DateTime','Lat','Lon','swDMS','SST', 'SAL']].dropna()
data_proc = PMEL.loc[:,['DateTime','Lat','Lon','swDMS','DMSPaq','DMSPp','DMSPt','sdepth']]

# Filter out data below 10 m
data_proc = data_proc[data_proc['sdepth']<10]

# Redefine columns as float data type to be readable by binning functions:
data_proc['DateTime'] = pd.to_datetime(data_proc['DateTime']).values.astype('float64')

#-----------------------------------------------------------------------------

#### Bin the data

# Bin data as averages across 1-m bins by sampling date:
# data_proc = data_proc.groupby(['DateTime', pd.cut(idx, bins)]).mean()
to_bin = lambda x: np.round(x /grid) * grid
data_proc['latbins'] = data_proc.Lat.map(to_bin)
data_proc['lonbins'] = data_proc.Lon.map(to_bin)
data_proc = data_proc.groupby(['DateTime', 'latbins', 'lonbins']).mean()

# Rename binned columns + drop mean lat/lons:
data_proc = data_proc.drop(columns=['Lat','Lon'])
data_proc = data_proc.rename_axis(index=['datetime', 'latbins', 'lonbins'])

# Transform dates back from integers to datetime numbers:
data_proc.reset_index(inplace=True) # remove index specification on columns
data_proc['datetime'] = pd.to_datetime(data_proc['datetime'],format=None)

# restrict date range
data_proc = data_proc[data_proc['datetime'].dt.strftime('%Y%m') > '1998-01']

months = pd.to_datetime(data_proc['datetime']).dt.strftime('%m').astype('float64')
# create a dict to call months in visualizations
var_months_ = {1:'January',
               2:'February',
               3:'March',
               4:'April',
               5:'May',
               6:'June',
               7:'July',
               8:'August',
               9:'September',
               10:'October',
               11:'November',
               12:'December'}
sizes = []
idx = np.arange(1,13,1)
for i in idx:
    sizes.append((months[months==i].shape[0]/months.shape[0])*100)

fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111)
ax.bar(idx,sizes)
ax.xaxis.set_ticks(idx)
ax.set_xticklabels([var_months_[i] for i in idx], rotation=90)
ax.set_ylabel('% Total Data')

# Filter to restrict only to certain months
data_proc = pd.concat([data_proc[(data_proc['datetime'].dt.month >= 10)],data_proc[(data_proc['datetime'].dt.month <= 4)]])

# Pull unique dates from data
unique_dates = np.unique(data_proc['datetime'].dt.strftime('%Y-%m'))
print(unique_dates)

#-----------------------------------------------------------------------------
#### Reshape and bin by month

# Pivot to move lat/lon pairs to columns - this is still at mins resolution temporally
# reset_index pulls the dates back into a column
DMS = data_proc.pivot(index='datetime',columns=['latbins','lonbins'], values='swDMS').reset_index()
DMSP = data_proc.pivot(index='datetime',columns=['latbins','lonbins'], values='DMSPt').reset_index()

# bin rows into months
DMS = DMS.groupby(DMS['datetime'].dt.strftime('%m')).mean()
DMSP = DMSP.groupby(DMSP['datetime'].dt.strftime('%m')).mean()

# stack as a column dataframe, perserving coordinates with NaNs
DMS = DMS.stack(['latbins'],dropna=False).stack(['lonbins'],dropna=False)
DMSP =  DMSP.stack(['latbins'],dropna=False).stack(['lonbins'],dropna=False)

#%% Load in Satellite Data
#-----------------------------------------------------------------------------

#### call the file directory where the CSV files are stored:
files_to_extract = sorted(os.listdir(write_dir))
#-----------------------------------------------------------------------------

#### create a dict, then dynmaically read in the variables
vars_ = dict()
for file in files_to_extract:
    print('loading: '+file.split('_')[2])
    # this extracts the variable name from the file name, and stores the data under it
    vars_[file.split('_')[2]] = pd.read_csv(write_dir+'/'+file,
                                            header=[0],
                                            index_col=[0,1,2]).squeeze('columns')
    # Make sure dates are floats for indexing
    vars_[file.split('_')[2]] = vars_[file.split('_')[2]].reset_index()
    vars_[file.split('_')[2]]['datetime'] = vars_[file.split('_')[2]]['datetime'].astype('float64')
    vars_[file.split('_')[2]] = vars_[file.split('_')[2]].set_index(['datetime','latbins','lonbins'])
    vars_[file.split('_')[2]] = vars_[file.split('_')[2]].squeeze('columns') # convert back to series

#### correct NPP - replace -9999 flag with nans
vars_['NPP'] = vars_['NPP'].replace(-9999.000,np.nan)

#-----------------------------------------------------------------------------

#### average the two wind speed climatologies together and add to dict
print()
print('loading wind climatologies')
#------------------------------------------------------------------------------
# Load CCMP
CCMP = pd.read_csv(write_dir[:69]+'/SO_DMS_wind_CCMP_0.18_deg.csv',
                   header=[0],
                   index_col=[0,1,2],).squeeze('columns')
#------------------------------------------------------------------------------
# Load Copernicus
Coper = pd.read_csv(write_dir[:69]+'/SO_DMS_wind_copernicus_0.18_deg.csv',
                   header=[0],
                   index_col=[0,1,2]).squeeze('columns').reindex_like(CCMP)
#------------------------------------------------------------------------------
# Average two estimates for each attribute
vars_['wind'] = pd.Series(np.nanmean([CCMP,Coper],axis=0), index=CCMP.index)

# Make sure dates are floats for indexing
# wind speed
vars_['wind'] = vars_['wind'].reset_index()
vars_['wind']['datetime'] = vars_['wind']['datetime'].astype('float64')
vars_['wind'] = vars_['wind'].set_index(['datetime','latbins','lonbins'])
vars_['wind'] = vars_['wind'].squeeze('columns') # convert back to series

del CCMP, Coper, #CCMP_U, CCMP_V, Coper_U, Coper_V

#-----------------------------------------------------------------------------
#### correct for length mismatch between variables

# to do this, need to first create a dummy series with rounded lat/lons - this 
# will be used in the final step to reindex (note: need to round coords for an exact match)
reindex_to_this = vars_['chl'].reset_index().copy()
reindex_to_this['latbins'] = reindex_to_this['latbins'].round(3)
reindex_to_this['lonbins'] = reindex_to_this['lonbins'].round(3)
reindex_to_this = reindex_to_this.set_index(['datetime','latbins','lonbins']).squeeze('columns')

# next loop through variables and correct index by adding/subtractiing the
# difference between lat/lons in the two indexes. Also round final index to
# remove rounding errors and obtain an exact match with the dummy series above.
# If a variables indices match but there is stil a length mismatch, can use the
# reindex_like() function directly.
print()
for var in vars_:
    if (var != 'chl') and (var != 'SSHA'):
        if vars_['chl'].index.levels[2][0]-vars_[var].index.levels[2][0] != 0 or vars_['chl'].index.levels[1][0]-vars_[var].index.levels[1][0] != 0:
            print('re-indexing '+var+'...')
            # extract difference in idx
            lon_corr = vars_['chl'].index.levels[2][0]-vars_[var].index.levels[2][0]
            lat_corr = vars_['chl'].index.levels[1][0]-vars_[var].index.levels[1][0]
            # reset idx and apply correction
            vars_[var] = vars_[var].reset_index()
            vars_[var]['latbins'] = vars_[var]['latbins']+lat_corr
            vars_[var]['lonbins'] = vars_[var]['lonbins']+lon_corr
            # round the idx for an exact match
            vars_[var]['latbins'] = vars_[var]['latbins'].round(3)
            vars_[var]['lonbins'] = vars_[var]['lonbins'].round(3)
            # convert back to indexed series 
            vars_[var] = vars_[var].set_index(['datetime','latbins','lonbins']).squeeze('columns')
            # finally, reindex
            vars_[var] = vars_[var].reindex_like(reindex_to_this)
        elif len(vars_[var]) != len(vars_['chl']):
            print('re-indexing '+var+'...')
            vars_[var] = vars_[var].reindex_like(vars_['chl'])
        else:
            pass
    elif var == 'SSHA':
        vars_[var] = vars_[var].reindex_like(vars_['chl'])

#-----------------------------------------------------------------------------

#### load etopo bathymetry data 
etopo = pd.read_csv(write_dir[:69]+'/'+'SO_etopo_0.18_deg.csv').set_index(['lonbins','latbins']).squeeze('columns')
frac_ocean = etopo[etopo<0].size/etopo.size
#-----------------------------------------------------------------------------

#### create a dict to call months in visualizations
var_months_ = {1:'January',
               2:'February',
               3:'March',
               4:'April',
               5:'May',
               6:'June',
               7:'July',
               8:'August',
               9:'September',
               10:'October',
               11:'November',
               12:'December'}

#%% Handle interpolated data

# first_run = False

if first_run is True:
    # interpolate the data
    #--------------------------------------------------------------------------
    # set up list and dict for data
    vars_interp = dict()
    # find number of variables
    num_vars = len(vars_)
    #-------------------------------------------------------------------------
    # loop through and interpolate variables
    for i, var in enumerate(vars_):
        print('Interpolating '+str(var)+'...'+' ('+str(i+1)+'/'+str(num_vars)+')')
        # find number of months
        num_months = len(np.unique(vars_[var].index.get_level_values('datetime')))
        single_var_interp = []
        if var == 'FSLE': # can comment this line out if we want to interpolate multiple variables at once
            if var == 'ice':
                # loop through months to interpolate
                for j, month in enumerate(np.unique(vars_[var].index.get_level_values('datetime').astype('int'))):
                    
                    # create 2 collumn array with lat/lon coordinates to interpolate
                    coords = np.stack([vars_[var].loc[month,:].index.get_level_values('lonbins').values,
                                       vars_[var].loc[month,:].index.get_level_values('latbins').values],axis=1)
                    
                    # index actual data and coordinates (i.e. remove nans)
                    ind = vars_[var].loc[month,:].notna() # filter out nans first
                    lon_pts = vars_[var].loc[month,:][ind].index.get_level_values('lonbins').values
                    lat_pts = vars_[var].loc[month,:][ind].index.get_level_values('latbins').values
                    values = vars_[var].loc[month,:][ind].values
                    
                    # interpolate data using a convex hull and linear function
                    interpd = scipy.interpolate.griddata(points=np.stack([lon_pts,lat_pts],axis=1),
                                                          values=values,
                                                          xi=coords,
                                                          method='linear')
                    
                    # create a series of the interpolated data
                    interpd = pd.Series(data=interpd, index=vars_[var].loc[month,:].index)
                    # append each month together
                    single_var_interp.append(interpd)
                    print('\t'+'Interpolated month: '+str(month)+' ('+str(j+1)+'/'+str(num_months)+' for '+var+')')
                # save our appended list as a single series inside the dict, and add dates back in
                vars_interp[var] = pd.Series(pd.concat(single_var_interp).values,index=vars_[var].index)
                # write this final data to file
                vars_interp[var].to_csv(write_dir[:69]+'/'+'SO_DMS_data_interp'+'/'+'SO_DMS_'+str(var)+'_'+str(round(grid,3))+'_deg_interp.csv')
            else:             
                # create 2 collumn array with lat/lon coordinates to interpolate
                coords = np.stack([vars_[var].index.get_level_values('datetime').values,
                                    vars_[var].index.get_level_values('latbins').values,
                                    vars_[var].index.get_level_values('lonbins').values],axis=1)
                
                # index actual data and coordinates (i.e. remove nans)
                ind = vars_[var].notna() # filter out nans first
                date_pts = vars_[var][ind].index.get_level_values('datetime').values
                lon_pts = vars_[var][ind].index.get_level_values('lonbins').values
                lat_pts = vars_[var][ind].index.get_level_values('latbins').values
                values = vars_[var][ind].values
                
                chunk_size = 100
                
                # interpolate
                interpd = scipy.interpolate.RBFInterpolator(da.from_array(np.stack([date_pts,lat_pts,lon_pts],axis=1),chunks=chunk_size),
                                                            da.from_array(values, chunks=chunk_size),
                                                            kernel='gaussian',
                                                            epsilon=2,
                                                            neighbors=50)(da.from_array(coords,chunks=chunk_size))
                                    
                # create a series of the interpolated data
                interpd = pd.Series(data=interpd, index=vars_[var].index)
                # Restrict interpolation to original data min/max bounds
                interpd.loc[interpd>vars_[var].max()] = vars_[var].max()
                interpd.loc[interpd<vars_[var].min()] = vars_[var].min()
                # save our appended list as a single series inside the dict, and add dates back in
                vars_interp[var] = interpd
                # write this final data to file
                vars_interp[var].to_csv(write_dir[:69]+'/'+'SO_DMS_data_interp'+'/'+'SO_DMS_'+str(var)+'_'+str(round(grid,3))+'_deg_interp.csv')
        else:
            pass
else:
    # set a dict to load data to
    vars_interp = dict()
    # call the file directory where the CSV files are stored:
    files_to_extract2 = sorted(os.listdir(write_dir[:69]+'/'+'SO_DMS_data_interp'))
    # load the previously interpolated data
    for file in files_to_extract2:
        print('loading: '+file.split('_')[2])
        # this extracts the variable name from the file name, and stores the data under it
        var = file.split('_')[2]
        vars_interp[var] = pd.read_csv(write_dir[:69]+'/'+'SO_DMS_data_interp'+'/'+file,
                                                       header=[0],
                                                       index_col=[0,1,2]).squeeze('columns')
        # **** To match reindexing above, round coords columns ****
        # Make sure dates are floats for indexing
        vars_interp[var] = vars_interp[var].reset_index()
        vars_interp[var]['datetime'] = vars_interp[var]['datetime'].astype('float64')
        vars_interp[var]['latbins'] = vars_interp[var]['latbins'].round(3)
        vars_interp[var]['lonbins'] = vars_interp[var]['lonbins'].round(3)
        vars_interp[var] = vars_interp[var].set_index(['datetime','latbins','lonbins']).squeeze('columns')
        vars_interp[var].name = var
        
    num_months = len(np.unique(vars_interp[file.split('_')[2]].index.get_level_values('datetime')))


#%% Add derived variables to the dict, filter out any bad interpolation

# Correct ice interpolation
vars_interp['ice'][vars_interp['ice'].isna()] = 0

# Correct for any erroneous interpolation
for var in vars_.keys():
    vars_interp[var].loc[vars_interp[var]>vars_[var].max()] = vars_[var].max()
    vars_interp[var].loc[vars_interp[var]<vars_[var].min()] = vars_[var].min()

# Si*, water mass tracer (see Sarmiento et al. 2003)
vars_interp['Si_star'] = vars_interp['Si']-vars_interp['SSN']

# NCP (algorithm from Li & Cassar 2016)
vars_interp['NCP'] = (8.57*vars_interp['NPP'])/(17.9+vars_interp['SST'])

# FLH:Chl-a, physiological stress indicator (see Westberry et al. 2013, 2019)
vars_interp['FLH_chl'] = vars_interp['FLH']/vars_interp['chl']

# Phi_cor, physiological stress indicator (see Behrenfield et al. 2009)
alpha = 0.0147*(vars_interp['chl']**-0.316)
vars_interp['phi_corr'] = (vars_interp['FLH']/(vars_interp['chl']*alpha*100))*(vars_interp['iPAR']/np.nanmean(vars_interp['iPAR']))

# Solar Radiatiion Dose (see Vallina & Simo, 2007)
vars_interp['SRD'] = (vars_interp['PAR']/(vars_interp['Kd']*vars_interp['MLD']))*(1-np.exp(-vars_interp['Kd']*vars_interp['MLD']))


#%% Select Training/Validation Data

#-----------------------------------------------------------------------------
#### reindex DMS data to match predictor data length (by adding in nans)

# pull the index into columns
DMS = DMS.reset_index()
# set datetime type for indexing
DMS['datetime'] = DMS['datetime'].astype('float64')
# round coords for exact match
DMS['latbins'] = DMS['latbins'].astype('float64').round(3)
DMS['lonbins'] = DMS['lonbins'].astype('float64').round(3)
# set index and squeeze back to series
DMS = DMS.set_index(['datetime','latbins','lonbins']).squeeze('columns')
# reindex to match predictors
DMS = DMS.reindex_like(reindex_to_this)

# rename as y variable:
y = DMS
y = y.rename('DMS')
#-----------------------------------------------------------------------------
#### concatenate predictors

# Select list of predictor variables to use in models
predictor_vars_ = [
    'CDOM',
    'MLD',
    'chl',
    'PAR',
    # 'SRD',
    'SAL',
    'SSHA',
    'SSN',
    'SST',
    'Si',
    'wind',
    # 'FLH_chl',
    # 'phi_corr',
    'ice',
    # 'FSLE',
    ]

# concatenate predictor variables into a matrix (coords x variable)
for i, var in tqdm(enumerate(predictor_vars_)):
    # print('concatenating: '+var)
    if i == 0:
        model_input = vars_interp[var].to_frame()
        model_input.columns = [var]
    else:
        model_input.insert(loc=i,column=var,value=vars_interp[var])
#-----------------------------------------------------------------------------

#### create dataframe for bathymetric data (need to replicate by num of months in dataset)
etopo = etopo.reset_index()
etopo['latbins'] = etopo['latbins'].round(3)
etopo['lonbins'] = etopo['lonbins'].round(3)
etopo = etopo.set_index(['latbins','lonbins']).squeeze('columns')
# need to reindex to match rest of data
bathy = pd.DataFrame(pd.concat([etopo.reindex_like(reindex_to_this.loc[1])]*num_months).values,
                     index=model_input.index)
bathy[bathy>0] = np.nan # filter out land values
# bathy[bathy==-1] = np.nan # bug in etopo data - filters out ice covered areas
bathy.columns = ['Bathymetry']

#-----------------------------------------------------------------------------

#### now setup our "X" (predictors) & "y" (DMS) to input into the models

X_full = pd.concat([y,bathy,model_input],axis=1,sort=False) # for final predictions
X = X_full.dropna().drop(['Bathymetry','DMS'], axis=1) # for training

# Training - now add in bathymetry to remove land/placeholder nans, then drop bathymetry
y = pd.concat([y,bathy],axis=1).dropna().drop(['Bathymetry'],axis=1).squeeze('columns')
# Training - apply IHS transformation to DMS data
y = np.arcsinh(y)
# finally, drop DMS first, remove nans associated with land, and the drop
# bathymetry from final predictor dataframe
X_full = X_full.drop(['DMS'],axis=1).dropna().drop(['Bathymetry'],axis=1)
#-----------------------------------------------------------------------------

#### split the data for training:testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print()
print('Proportion of training data = %.2f' % round(X_train.shape[0]/X.shape[0],2))
print('Proportion of testing data = %.2f' % round(X_test.shape[0]/X.shape[0],2))
#-----------------------------------------------------------------------------

#### standardize the data
scaler = StandardScaler()
scaler.fit(X_train)
X_train = pd.DataFrame(scaler.transform(X_train), index=X_train.index, columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)

#-----------------------------------------------------------------------------
# clear some memory
# del bathy, etopo, model_input, DMS, vars_
del bathy, model_input, vars_

#%% Generate land/ice mask

# import the shapefiles - these are the files containing polygons for ice/land masses
glaciers = shpreader.Reader('C:/Users/bcamc/OneDrive/Desktop/Python/Projects/sulfur/southern_ocean/ice_shapefiles/ne_10m_glaciated_areas/ne_10m_glaciated_areas.shp')
ice_shelves = shpreader.Reader('C:/Users/bcamc/OneDrive/Desktop/Python/Projects/sulfur/southern_ocean/ice_shapefiles/ne_10m_antarctic_ice_shelves_polys/ne_10m_antarctic_ice_shelves_polys.shp')
land_shp = shpreader.Reader(shpreader.natural_earth(resolution='50m',category='physical', name='land'))

# Prep the geometry files - this combines ice and land polygons + speeds up iterative speed later
shapes = [land_shp, glaciers, ice_shelves]
geoms = unary_union([unary_union(list(i.geometries())) for i in shapes])
land = prep(geoms)

# Generate the mask - use shapely's functions to check whether each coordinate is within a land/ice polygon
# note: shapely.vectorized will significantly reduce computational times
to_mask = np.empty(len(X_full.loc[1,'chl']))
print(len(X_full.loc[1,'chl']))
for ind,(i,j) in tqdm(enumerate(zip(X_full.loc[1,'chl'].index.get_level_values('lonbins').values, X_full.loc[1,'chl'].index.get_level_values('latbins').values))):
    to_mask[ind] = land.contains(sgeom.Point(i,j))
# turn the array into a series with indexed coordinates
to_mask = pd.Series(np.tile(to_mask, len(X_full.loc[:,'chl'].index.levels[0])), index=X_full.loc[:,'chl'].index)
# replace 1.0 (i.e land/ice) with nans to mask data
to_mask = to_mask.replace(1.0,np.nan)

#%% Mask out the full dataset for predictions

X_full = X_full.where(to_mask.notna(),np.nan).dropna()

#%% Iron-stress proxy testing

# first_run = True

# Set directories to store ANN ensembles
if 'FLH_chl' in X_full.columns.values:
    os.chdir(dir_new)
if 'phi_corr' in X_full.columns.values:
    os.chdir(dir_new)

#%% ANN - Build Artifical Neural Network ensemble

ensemble_start = timeit.default_timer()

#------------------------------------------------------------------------------
#### Train/fit or load ensemble

logger = set_logger('ensemble_test', use_tb_logger=True)

# Initiate ensemble
ANN_ensemble = torchensemble.VotingRegressor(estimator=ANNRegressor(nfeatures=X_train.shape[1]),
                                            n_estimators=1000,
                                            cuda=False,
                                            n_jobs=-1,
                                            )

#### Set our optimizer function & dynamic learning rate algorithm
ANN_ensemble.set_optimizer('Adam',
                        lr=1e-1,
                        weight_decay=1e-6,
                        eps=1e-9)
ANN_ensemble.set_scheduler('StepLR',
                        step_size=30,
                        gamma=0.1)

#### train or load ensemble
if first_run is True:
    #### Process Data into Tensors
    #------------------------------------------------------------------------------
    X_valtrain, X_val, y_valtrain, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)
    
    # compile tensors together into datasets
    train_dataset = TensorDataset(torch.from_numpy(X_valtrain.values).float(), torch.from_numpy(y_valtrain.values).view(len(y_valtrain.values),1).float())
    test_dataset = TensorDataset(torch.from_numpy(X_test.values).float(), torch.from_numpy(y_test.values).view(len(y_test.values),1).float())
    val_dataset = TensorDataset(torch.from_numpy(X_val.values).float(), torch.from_numpy(y_val.values).view(len(y_val.values),1).float())
    del X_valtrain, y_valtrain, X_val, y_val
    
    # build dataloaders
    batch_size=100
    train_dataloader = DataLoader(
                            dataset=train_dataset, 
                            batch_size=batch_size, 
                            shuffle=True,
                            num_workers=0,
                            )
    test_dataloader = DataLoader(
                        dataset=test_dataset, 
                        batch_size=batch_size, 
                        shuffle=True,
                        num_workers=0,
                        )
    val_dataloader = DataLoader(
                        dataset=val_dataset, 
                        batch_size=batch_size, 
                        shuffle=True,
                        num_workers=0,
                        )
    #### Fit ensemble
    ANN_ensemble.fit(train_loader=train_dataloader,
                  epochs=100,
                  test_loader=val_dataloader,
                  save_model=True,
                  # early_stopping_rounds=2,
              )
    #### get final MSE
    mse = ANN_ensemble.evaluate(test_loader=test_dataloader)
    print(f'MSE loss: {mse:.2f}')
    # save model:
    # io.save(model, save_dir=ANN_save_dir, logger=logger) # don't need if fit() calls save param
else:
    io.load(ANN_ensemble, save_dir=ANN_save_dir)

#### Get accuracy of final ensemble
print('Rendering training predictions...')
ANN_y_train_pred = ANN_ensemble.predict(torch.from_numpy(X_train.values).float()).detach().numpy()
print('Rendering testing predictions...')
ANN_y_test_pred = ANN_ensemble.predict(torch.from_numpy(X_test.values).float()).detach().numpy()
ANN_train_R2 = r2_score(y_train, ANN_y_train_pred)
ANN_ensemble_R2 = r2_score(y_test, ANN_y_test_pred)
print(f"Ensemble training accuracy: {ANN_train_R2*100:.2f}%")
print(f"Accuracy of ensemble: {ANN_ensemble_R2*100:.2f}%")

#### Run full predictions
print('Rendering final SO predictions...')
to_predict  = torch.from_numpy(scaler.transform(X_full)).float()
ANN_y_pred = ANN_ensemble.predict(to_predict).detach().numpy()
ANN_y_pred = pd.Series(ANN_y_pred[:,0], index=X_full.index)

#-----------------------------------------------------------------------------
# Calculate stds, pearson correlations, and RMSEs for members in ANN ensemble:
ANN_stds = np.std([ANN_ensemble[model].predict(X_test.values) for model in tqdm(range(len(ANN_ensemble)))],axis=1)

ANN_corrcoefs = np.empty([len(ANN_ensemble)])
for i, model in tqdm(enumerate(ANN_ensemble)):
    rs = pearsonr(ANN_ensemble[i].predict(X_test.values), y_test.values)
    ANN_corrcoefs[i] = rs[0]

ANN_rmses = np.empty([len(ANN_ensemble)])
for i, model in tqdm(enumerate(ANN_ensemble)):
    ANN_rmses[i] = np.sqrt(metrics.mean_squared_error(y_test, ANN_ensemble[i].predict(X_test.values)))

#-----------------------------------------------------------------------------
# Get runtime
ensemble_end= timeit.default_timer()
ANN_execution_time = ensemble_end-ensemble_start
#-----------------------------------------------------------------------------
#### Evaluate the model
print()
print('~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~')
print('       PyTorch ANN Model Results       ')
print('~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~')
print('\nExecution time:')
print(str(round(ANN_execution_time,5)),'seconds') 
print(str(round((ANN_execution_time)/60,5)),'mins')
print(str(round((ANN_execution_time)/3600,5)),'hrs')
print()
print('Model Configuration:')
print(ANN_ensemble[0])
print()
print('Number of models in ensemble:',str(len(ANN_ensemble)))
print('Mean Absolute Error (MAE):', round(metrics.mean_absolute_error(y_test, ANN_y_test_pred),4))
print('Mean Squared Error (MSE):', round(metrics.mean_squared_error(y_test, ANN_y_test_pred),4))
print('Root Mean Squared Error (RMSE):', round(np.sqrt(metrics.mean_squared_error(y_test, ANN_y_test_pred)),4))
# print('Mean Prediction Accuracy (100-MAPE):', round(accuracy, 2), '%')
ANN_ensemble_R2 = r2_score(y_test,ANN_y_test_pred)
print('Training Root Mean Squared Error (RMSE):', round(np.sqrt(metrics.mean_squared_error(y_train, ANN_y_train_pred)),4))
print("Training accuracy (R^2): %0.3f" % ANN_train_R2)
print('Testing Root Mean Squared Error (RMSE):', round(np.sqrt(metrics.mean_squared_error(y_test, ANN_y_test_pred)),4))
print("Testing accuracy (R^2): %0.3f" % ANN_ensemble_R2)
print('~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~')
print()
#------------------------------------------------------------------------------
#%% RFR - Build RFR model
#-----------------------------------------------------------------------------
#### Define model
start = timeit.default_timer() # start the clock

nfeatures = np.min(X_train.shape)

RFR_model = RandomForestRegressor(n_estimators=1000,
                                  max_features=nfeatures,
                                  min_samples_leaf=1, # 1 is default
                                  max_depth=25, # None is default
                                  # ccp_alpha=0, # 0 is default
                                  n_jobs=-1, # use all core processors in computer (i.e. speed up computation)
                                  random_state=0,# this just seeds the randomization of the ensemble models each time
                                  bootstrap=True,
                                  oob_score=False,
                                  verbose=False) 

# fit the model to the training data
RFR_model.fit(X_train.values, y_train.values)
#-----------------------------------------------------------------------------
#### Validate the model
RFR_y_pred_test = RFR_model.predict(X_test.values)

n_features = RFR_model.n_features_in_

RFR_model_R2 = RFR_model.score(X_test.values,y_test.values)
#-----------------------------------------------------------------------------
#### Model prediction of DMS values
RFR_y_pred = RFR_model.predict(scaler.transform(X_full)) 
RFR_y_pred = pd.Series(RFR_y_pred,index=X_full.index, name='DMS')
#-----------------------------------------------------------------------------
#### Calcuate stds, pearson correlations for RFR trees in ensemble:

RFR_stds = np.std([single_tree.predict(X_test.values) for single_tree in tqdm(RFR_model.estimators_)],axis=1)

RFR_corrcoefs = np.empty([len(RFR_model.estimators_)])
for i, single_tree in tqdm(enumerate(RFR_model.estimators_)):
    rs = pearsonr(single_tree.predict(X_test.values), y_test.values)
    RFR_corrcoefs[i] = rs[0]

RFR_rmses = np.empty([len(RFR_model.estimators_)])
for i, single_tree in tqdm(enumerate(RFR_model.estimators_)):
    RFR_rmses[i] = np.sqrt(metrics.mean_squared_error(y_test, single_tree.predict(X_test.values)))


#-----------------------------------------------------------------------------
end = timeit.default_timer() # stop the clock
#### Evaluate the model
print()
print('~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~')
print('             RFR Model Results         ')
print('~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~')
RFR_execution_time = end-start
print('\nExecution time:')
print(str(round(RFR_execution_time,5)),'seconds') 
print(str(round((RFR_execution_time)/60,5)),'mins')
print(str(round((RFR_execution_time)/3600,5)),'hrs')
print('Number of trees in ensemble:',str(RFR_model.n_estimators))
print('Mean Absolute Error (MAE):', round(metrics.mean_absolute_error(y_test.values, RFR_y_pred_test),4))
print('Mean Squared Error (MSE):', round(metrics.mean_squared_error(y_test.values, RFR_y_pred_test),4))
print('Training Root Mean Squared Error (RMSE):', round(np.sqrt(metrics.mean_squared_error(y_train.values, RFR_model.predict(X_train.values))),4))
print("Training accuracy (R^2): %0.3f" % RFR_model.score(X_train.values, y_train.values))
print('Testing Root Mean Squared Error (RMSE):', round(np.sqrt(metrics.mean_squared_error(y_test.values, RFR_y_pred_test)),4))
print("Testing accuracy (R^2): %0.3f" % RFR_model.score(X_test.values, y_test.values))
print('- - - - - - - - - - - -')
print('Full model R^2: %0.3f' % RFR_model_R2)
print('Full model RMSE: %0.3f' % np.sqrt(metrics.mean_squared_error(y_test.values, RFR_y_pred_test)))
print('~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~')
print()
#------------------------------------------------------------------------------

#%% Combine both model's predictions
if first_run == True:
    # Load in data
    export_dir = 'C:/Users/bcamc/OneDrive/Desktop/Python/Projects/sulfur/southern_ocean/export_data/'
    #-----------------------------------------------------------------------------
    # Complie model predictions together
    models_combined = pd.Series(np.nanmean(pd.concat([np.sinh(RFR_y_pred), np.sinh(ANN_y_pred)], axis=1),axis=1), index=RFR_y_pred.index, name='DMS')
    #-----------------------------------------------------------------------------
    # Create indices to subsample from full dataset
    import random
    random.seed(0) # for reproducibility
    ind = random.sample(range(0,len(X_full)),5000)
    #-----------------------------------------------------------------------------
    # Compute SSHA gradients
    var = 'SSHA'
    grads=[]
    for i in X_full.index.levels[0].values:
        # Calculate dx & dy from the matrix of the variable for a single month
        gradient = np.gradient(X_full.loc[i,var].unstack('lonbins'))
        # Calculate the resultant vector
        gradient = np.sqrt(gradient[0]**2+gradient[1]**2)
        # Create a dataframe and stack back to a series
        gradient = pd.DataFrame(gradient, 
                                index=X_full.loc[i,var].unstack('lonbins').index,
                                columns=X_full.loc[i,var].unstack('lonbins').columns).stack()
        # Add back in a date index
        gradient = gradient.reindex_like(X_full.loc[i,var])
        gradient = gradient.reset_index()
        gradient['datetime'] = np.tile(i,len(gradient))
        gradient = gradient.set_index(['datetime','latbins','lonbins'])
        gradient = gradient.squeeze('columns')
        # Append final derivatives to list
        grads.append(gradient)
    dSSHA = pd.concat(grads)
    var = 'SST'
    grads=[]
    for i in X_full.index.levels[0].values:
        # Calculate dx & dy from the matrix of the variable for a single month
        gradient = np.gradient(X_full.loc[i,var].unstack('lonbins'))
        # Calculate the resultant vector
        gradient = np.sqrt(gradient[0]**2+gradient[1]**2)
        # Create a dataframe and stack back to a series
        gradient = pd.DataFrame(gradient, 
                                index=X_full.loc[i,var].unstack('lonbins').index,
                                columns=X_full.loc[i,var].unstack('lonbins').columns).stack()
        # Add back in a date index
        gradient = gradient.reindex_like(X_full.loc[i,var])
        gradient = gradient.reset_index()
        gradient['datetime'] = np.tile(i,len(gradient))
        gradient = gradient.set_index(['datetime','latbins','lonbins'])
        gradient = gradient.squeeze('columns')
        # Append final derivatives to list
        grads.append(gradient)
    dSST = pd.concat(grads)
    del gradient, grads
    #------------------------------------------------------------------------------
    # Add in new predictors to run GP on
    X_full_plus = X_full.copy()
    X_full_plus['dSSHA'] = dSSHA.fillna(0)
    X_full_plus['SRD'] = vars_interp['SRD'].reindex_like(X_full.loc[:,'chl'])
    X_full_plus['currents'] = vars_interp['currents'].reindex_like(X_full.loc[:,'chl'])
    #------------------------------------------------------------------------------
    # Write data to files for easier test handling later
    ANN_y_pred.to_csv(export_dir+'ANN_y_pred.csv')
    RFR_y_pred.to_csv(export_dir+'RFR_y_pred.csv')
    models_combined.to_csv(export_dir+'models_combined.csv')
    y.to_csv(export_dir+'y.csv')
    X.to_csv(export_dir+'X.csv')
    X_full_plus.to_csv(export_dir+'X_full_plus.csv')

    # to test with the Eureqa software
    # X_full_plus.iloc[ind].to_csv(export_dir+'X_full_plus_subsample.csv')
    # models_combined.iloc[ind].to_csv(export_dir+'models_combined_subset.csv')

#%% Calculate build run time
analysis_end = timeit.default_timer()
analysis_runtime = analysis_end-analysis_start
print('Analysis Runtime:')
print(str(round(analysis_runtime,5)),'secs')
print(str(round((analysis_runtime)/60,5)),'mins')
print(str(round((analysis_runtime)/3600,5)),'hrs')
