# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 08:06:31 2021

@author: bcamc
"""

#%% Import Packages
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition, inset_axes
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np
import scipy
from scipy.stats.stats import pearsonr, spearmanr
import cartopy
import cartopy.crs as ccrs
from mpl_toolkits.axes_grid1 import make_axes_locatable
import xarray as xr
from sklearn.decomposition import IncrementalPCA
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from datetime import datetime
from sklearn import linear_model
import datetime
import cartopy.io.shapereader as shpreader
from cartopy.feature import ShapelyFeature
import shapely.geometry as sgeom
from shapely.ops import unary_union
from shapely.prepared import prep
import joblib
from joblib import Parallel, delayed 
from obspy.geodetics import kilometers2degrees, degrees2kilometers
import cmocean
import seaborn as sns
from tabulate import tabulate

# Progress bar package
from tqdm import tqdm

# Gibbs seawater properties packages
import gsw

# Import pre-built mapping functions
from SO_mapping_templates import haversine, South_1ax_map, South_1ax_flat_map
# Import function to calculate fluxes 
from Fluxes import calculate_fluxes
# Import taylor diagram script
from taylorDiagram import TaylorDiagram

#%% Define directories

front_dir = 'C:/Users/bcamc/OneDrive/Desktop/Python/Projects/sulfur/southern_ocean/'
lana_dir = 'C:/Users/bcamc/OneDrive/Desktop/Python/Projects/sulfur/southern_ocean/dmsclimatology/'
jarnikova_dir = 'C:/Users/bcamc/OneDrive/Desktop/Python/Projects/sulfur/southern_ocean/Jarnikova_SO_files/'

#%% Set working directories
dir_ = 'C:\\Users\\bcamc\\OneDrive\\Desktop\\Python\\Projects\\sulfur\\southern_ocean\\Scripts'
if os.getcwd() != dir_:
    os.chdir(dir_)
    
#%% Read in data (optional)

export_dir = 'C:/Users/bcamc/OneDrive/Desktop/Python/Projects/sulfur/southern_ocean/export_data/'
models_combined = pd.read_csv(export_dir+'models_combined.csv').set_index(['datetime','latbins','lonbins']).squeeze('columns')
X_full_plus = pd.read_csv(export_dir+'X_full_plus.csv').set_index(['datetime','latbins','lonbins'])


# ANN_y_pred = pd.read_csv(export_dir+'ANN_y_pred.csv').set_index(['datetime','latbins','lonbins']).squeeze('columns')
# RFR_y_pred = pd.read_csv(export_dir+'RFR_y_pred.csv').set_index(['datetime','latbins','lonbins']).squeeze('columns')
# y = pd.read_csv(export_dir+'y.csv').set_index(['datetime','latbins','lonbins']).squeeze('columns')
# X = pd.read_csv(export_dir+'X.csv').set_index(['datetime','latbins','lonbins'])
# X_full = X_full_plus.drop(['dSSHA','currents','SRD'],axis=1)

#%% Post-processing

# ***** Load in models/data using "SO_DMS_build_models.py" *****

# for plotting
reordered_months = np.array([10.,11.,12.,1.,2.,3.,4.])

# Average predictions
RFR_y_pred_mean = np.sinh(RFR_y_pred).groupby(['latbins','lonbins']).mean()
ANN_y_pred_mean = np.sinh(ANN_y_pred).groupby(['latbins','lonbins']).mean()

# calculate Si*
Si_star = (X_full.loc[:,'Si']-X_full.loc[:,'SSN']).squeeze()
X_full_plus['Si_star'] = Si_star
#------------------------------------------------------------------------------
# Import ACC front locations
front_data = xr.open_dataset(front_dir+'Park_durand_fronts.nc')
fronts = dict()
to_bin = lambda x: np.round(x /grid) * grid
#------------------------------------------------------------------------------
# NB front
fronts['NB'] = pd.DataFrame(np.stack([front_data.LatNB.values,
                            front_data.LonNB.values,
                            np.ones(front_data.LonNB.values.shape)],axis=1),
                  columns=['latbins','lonbins','locs'])
fronts['NB'] = fronts['NB'].sort_values('lonbins')
fronts['NB']['latbins'] = fronts['NB']['latbins'].map(to_bin).round(3)
fronts['NB']['lonbins'] = fronts['NB']['lonbins'].map(to_bin).round(3)
fronts['NB'] = fronts['NB'].set_index(['latbins','lonbins']).squeeze()
fronts['NB'] = fronts['NB'][~fronts['NB'].index.duplicated(keep='first')]
# fronts['NB'] = fronts['NB'].reindex_like(models_combined.loc[1])
#------------------------------------------------------------------------------
# SAF front
fronts['SAF'] = pd.DataFrame(np.stack([front_data.LatSAF.values,
                            front_data.LonSAF.values,
                            np.ones(front_data.LonSAF.values.shape)],axis=1),
                  columns=['latbins','lonbins','locs'])
fronts['SAF'] = fronts['SAF'].sort_values('lonbins')
fronts['SAF']['latbins'] = fronts['SAF']['latbins'].map(to_bin).round(3)
fronts['SAF']['lonbins'] = fronts['SAF']['lonbins'].map(to_bin).round(3)
fronts['SAF'] = fronts['SAF'].set_index(['latbins','lonbins']).squeeze()
fronts['SAF'] = fronts['SAF'][~fronts['SAF'].index.duplicated(keep='first')]
# fronts['SAF'] = fronts['SAF'].reindex_like(models_combined.loc[1])
#------------------------------------------------------------------------------
# PF front
fronts['PF'] = pd.DataFrame(np.stack([front_data.LatPF.values,
                            front_data.LonPF.values,
                            np.ones(front_data.LonPF.values.shape)],axis=1),
                  columns=['latbins','lonbins','locs'])
fronts['PF'] = fronts['PF'].sort_values('lonbins')
fronts['PF']['latbins'] = fronts['PF']['latbins'].map(to_bin).round(3)
fronts['PF']['lonbins'] = fronts['PF']['lonbins'].map(to_bin).round(3)
fronts['PF'] = fronts['PF'].set_index(['latbins','lonbins']).squeeze()
fronts['PF'] = fronts['PF'][~fronts['PF'].index.duplicated(keep='first')]
# fronts['PF'] = fronts['PF'].reindex_like(models_combined.loc[1])
#------------------------------------------------------------------------------
# SACCF front
fronts['SACCF'] = pd.DataFrame(np.stack([front_data.LatSACCF.values,
                            front_data.LonSACCF.values,
                            np.ones(front_data.LonSACCF.values.shape)],axis=1),
                  columns=['latbins','lonbins','locs'])
fronts['SACCF'] = fronts['SACCF'].sort_values('lonbins')
fronts['SACCF']['latbins'] = fronts['SACCF']['latbins'].map(to_bin).round(3)
fronts['SACCF']['lonbins'] = fronts['SACCF']['lonbins'].map(to_bin).round(3)
fronts['SACCF'] = fronts['SACCF'].set_index(['latbins','lonbins']).squeeze()
fronts['SACCF'] = fronts['SACCF'][~fronts['SACCF'].index.duplicated(keep='first')]
# fronts['SACCF'] = fronts['SACCF'].reindex_like(models_combined.loc[1])
#------------------------------------------------------------------------------
# SB front
fronts['SB'] = pd.DataFrame(np.stack([front_data.LatSB.values,
                            front_data.LonSB.values,
                            np.ones(front_data.LonSB.values.shape)],axis=1),
                  columns=['latbins','lonbins','locs'])
fronts['SB'] = fronts['SB'].sort_values('lonbins')
fronts['SB']['latbins'] = fronts['SB']['latbins'].map(to_bin).round(3)
fronts['SB']['lonbins'] = fronts['SB']['lonbins'].map(to_bin).round(3)
fronts['SB'] = fronts['SB'].set_index(['latbins','lonbins']).squeeze()
fronts['SB'] = fronts['SB'][~fronts['SB'].index.duplicated(keep='first')]
# fronts['SB'] = fronts['SB'].reindex_like(models_combined.loc[1])

# front_data.close(); del front_data
#------------------------------------------------------------------------------
SA = gsw.SA_from_SP(SP=X_full.loc[:,'SAL'].values, p=1, lon=X_full.index.get_level_values('lonbins').values, lat=X_full.index.get_level_values('latbins').values)
CT = gsw.CT_from_t(SA=SA, t=X_full.loc[:,'SST'].values, p=1)
density = gsw.density.rho(SA=SA,CT=CT,p=1)
density = pd.Series(density, index=X_full.loc[:,'chl'].index)

#%% Model Sea-Air Fluxes
#-----------------------------------------------------------------------------
#~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
#                          ===================
#                               RFR Model
#                          ===================
#-----------------------------------------------------------------------------
# Fluxes (umol m^-2 d^-1):
RFR_flux = dict()
k_dms, RFR_flux['GM12'] = calculate_fluxes(data=np.sinh(RFR_y_pred).values,
                                        ice_cover=X_full.loc[:,'ice'].values,
                                        wind_speed=X_full.loc[:,'wind'].values,
                                        T=X_full.loc[:,'SST'].values,
                                        parameterization='GM12')
_, RFR_flux['SD02'] = calculate_fluxes(data=np.sinh(RFR_y_pred).values,
                                        ice_cover=X_full.loc[:,'ice'].values,
                                        wind_speed=X_full.loc[:,'wind'].values,
                                        T=X_full.loc[:,'SST'].values,
                                        parameterization='SD02')
# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
# Set as series
RFR_flux['GM12'] = pd.Series(RFR_flux['GM12'], index=X_full.loc[:,'SST'].index, name='DMS flux')
# filter out negative estimates
RFR_flux['GM12'] = RFR_flux['GM12'][(RFR_flux['GM12'] >= 0) & (RFR_flux['GM12'].notna())].reindex_like(RFR_y_pred)
RFR_flux['SD02'] = pd.Series(RFR_flux['SD02'], index=X_full.loc[:,'SST'].index, name='DMS flux')
#-----------------------------------------------------------------------------
#                          ===================
#                               ANN Model
#                          ===================
#-----------------------------------------------------------------------------
ANN_flux = dict()
_, ANN_flux['GM12'] = calculate_fluxes(data=np.sinh(ANN_y_pred).values,
                                        ice_cover=X_full.loc[:,'ice'].values,
                                        wind_speed=X_full.loc[:,'wind'].values,
                                        T=X_full.loc[:,'SST'].values,
                                        parameterization='GM12')
_, ANN_flux['SD02'] = calculate_fluxes(data=np.sinh(ANN_y_pred).values,
                                        ice_cover=X_full.loc[:,'ice'].values,
                                        wind_speed=X_full.loc[:,'wind'].values,
                                        T=X_full.loc[:,'SST'].values,
                                        parameterization='SD02')
# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
# Set as series
ANN_flux['GM12'] = pd.Series(ANN_flux['GM12'], index=X_full.loc[:,'SST'].index, name='DMS flux')
# filter out negative estimates
ANN_flux['GM12'] = ANN_flux['GM12'][(ANN_flux['GM12'] >= 0) & (ANN_flux['GM12'].notna())].reindex_like(ANN_y_pred)
ANN_flux['SD02'] = pd.Series(ANN_flux['SD02'], index=X_full.loc[:,'SST'].index, name='DMS flux')
#-----------------------------------------------------------------------------
#                          ===================
#                                Actual
#                          ===================
#-----------------------------------------------------------------------------
obs_flux = dict()
_, obs_flux['GM12'] = calculate_fluxes(data=np.sinh(y).values,
                                        ice_cover=X.loc[:,'ice'].values,
                                        wind_speed=X.loc[:,'wind'].values,
                                        T=X.loc[:,'SST'].values,
                                        parameterization='GM12')
_, obs_flux['SD02'] = calculate_fluxes(data=np.sinh(y).values,
                                        ice_cover=X.loc[:,'ice'].values,
                                        wind_speed=X.loc[:,'wind'].values,
                                        T=X.loc[:,'SST'].values,
                                        parameterization='SD02')
# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
# Set as series
obs_flux['GM12'] = pd.Series(obs_flux['GM12'], index=X.loc[:,'SST'].index, name='DMS flux')
# filter out negative estimates
obs_flux['GM12'] = obs_flux['GM12'][(obs_flux['GM12'] >= 0) & (obs_flux['GM12'].notna())].reindex_like(y)
obs_flux['SD02'] = pd.Series(obs_flux['SD02'], index=X.loc[:,'SST'].index, name='DMS flux')
#-----------------------------------------------------------------------------
#                          ===================
#                            Regional Fluxes
#                          ===================
#-----------------------------------------------------------------------------
#~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
# Constants:
A = ((max_lat-min_lat)*111*1000)*((max_lon-min_lon)*111*1000) # total regional area
A_ocean = A*frac_ocean # fraction of total area covered by ocean
S_mol_mass = 32.06 # molar mass of sulfur
num_days = np.sum(np.array([31,30,31,31,28,31,30])) # number of total days in the dataset
#-----------------------------------------------------------------------------

# Regional modelled flux (convert to Tg over total days)

RFR_flux_reg = (RFR_flux['GM12']*S_mol_mass*A_ocean*num_days)/(1e6*1e12)
ANN_flux_reg = (ANN_flux['GM12']*S_mol_mass*A_ocean*num_days)/(1e6*1e12)
obs_flux_reg = (obs_flux['GM12']*S_mol_mass*A_ocean*num_days)/(1e6*1e12)

fluxes_combined = pd.concat([RFR_flux['GM12'], ANN_flux['GM12']], axis=1).mean(axis=1)

#%% Lana Climatology Sea-air Fluxes
files = os.listdir(lana_dir)

# Set 1x1o coords
lana_coords = dict()
lana_coords['lat'] = pd.Series(np.arange(-89,91,1), name='latbins')
lana_coords['lon'] = pd.Series(np.arange(-179,181,1), name='lonbins')
time_match = {'OCT':10,'NOV':11,'DEC':12,'JAN':1,'FEB':2,'MAR':3,'APR':4}

# Retrive DMS climatology values, adding lats/lons to dataframes
lana_clim = []
for file in files:
    frame = pd.DataFrame(np.flipud(pd.read_csv(lana_dir+file, header=None)),
                            index=lana_coords['lat'], columns=lana_coords['lon'])
    frame = frame.stack(dropna=False)
    frame = frame.reset_index()
    frame['datetime'] = np.tile(float(time_match[file.split('.')[0][-3:]]), len(frame))
    frame = frame.set_index(['datetime','latbins','lonbins']).squeeze()
    frame.name = 'DMS'
    lana_clim.append(frame)
lana_clim = pd.concat(lana_clim)

# Regrid variables to compute sea-air fluxes
lana = dict()
for var in ['wind','ice','SST']:
    lana[var] = X_full.loc[:,var].copy()
    lana[var] = lana[var].reset_index()
    lana[var] = lana[var].rename(columns={'lonbins':'lon','latbins':'lat'})
    
    # regrid to nearest degree (i.e. 1x1o grid)
    lana[var]['latbins'] = lana[var].lat.round(0).astype('int32')
    lana[var]['lonbins'] = lana[var].lon.round(0).astype('int32')
    lana[var] = lana[var].set_index(['datetime','latbins','lonbins'])
    lana[var] = lana[var].drop(columns=['lat','lon'])
    lana[var] = lana[var].groupby(['datetime','latbins','lonbins']).mean().squeeze()
    lana[var] = lana[var].sort_index().reindex_like(lana_clim)
    print(var+' regrid complete')

# Compute sea-air flux
#-----------------------------------------------------------------------------
lana_flux = dict()
_, lana_flux['GM12'] = calculate_fluxes(data=lana_clim.values,
                                        ice_cover=lana['ice'].values,
                                        wind_speed=lana['wind'].values,
                                        T=lana['SST'].values,
                                        parameterization='GM12')
_, lana_flux['SD02'] = calculate_fluxes(data=lana_clim.values,
                                        ice_cover=lana['ice'].values,
                                        wind_speed=lana['wind'].values,
                                        T=lana['SST'].values,
                                        parameterization='SD02')
# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
# Set as series
lana_flux['GM12'] = pd.Series(lana_flux['GM12'], index=lana['SST'].index, name='DMS flux')
# filter out negative estimates
lana_flux['GM12'] = lana_flux['GM12'][(lana_flux['GM12'] >= 0) & (lana_flux['GM12'].notna())].reindex_like(lana_clim)
lana_flux['SD02'] = pd.Series(lana_flux['SD02'], index=lana['SST'].index, name='DMS flux')
#-----------------------------------------------------------------------------
del frame

#%% Jarnikova Climatology Sea-air Fluxes

# This climatology is from Dec to Feb (Jarnikova & Tortell, 2016)
mat = scipy.io.loadmat(jarnikova_dir+'nov26product.mat')

tj_dms = mat['structname'][0,1]['barnessmooth'][0,0]
tj_lats = mat['structname'][0,1]['latvec'][0,0][0,:]
tj_lons = mat['structname'][0,1]['lonvec'][0,0][0,:]

jarnikova_clim = pd.DataFrame(tj_dms, index=tj_lats, columns=tj_lons)
jarnikova_clim.index = jarnikova_clim.index.rename('latbins')
jarnikova_clim.columns = jarnikova_clim.columns.rename('lonbins')
jarnikova_clim = jarnikova_clim.stack()

# Reindex like lana et al. climatology
jarnikova_clim = jarnikova_clim.reindex_like(lana_clim.loc[[12,1,2]].groupby(['latbins','lonbins']).mean())

# Calculate the fluxes
#-----------------------------------------------------------------------------
jarnikova_flux = dict()
_, jarnikova_flux['GM12'] = calculate_fluxes(data=jarnikova_clim,
                                     ice_cover=lana['ice'].loc[[12,1,2]].groupby(['latbins','lonbins']).mean(),
                                     wind_speed=lana['wind'].loc[[12,1,2]].groupby(['latbins','lonbins']).mean(),
                                     T=lana['SST'].loc[[12,1,2]].groupby(['latbins','lonbins']).mean(),
                                     parameterization='GM12')
_, jarnikova_flux['SD02'] = calculate_fluxes(data=jarnikova_clim.values,
                                     ice_cover=lana['ice'].loc[[12,1,2]].groupby(['latbins','lonbins']).mean().values,
                                     wind_speed=lana['wind'].loc[[12,1,2]].groupby(['latbins','lonbins']).mean().values,
                                     T=lana['SST'].loc[[12,1,2]].groupby(['latbins','lonbins']).mean().values,
                                     parameterization='SD02')
# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
# Set as series
jarnikova_flux['GM12'] = pd.Series(jarnikova_flux['GM12'], index=lana['SST'].loc[[12,1,2]].groupby(['latbins','lonbins']).mean().index, name='DMS flux')
# filter out negative estimates
jarnikova_flux['GM12'] = jarnikova_flux['GM12'][(jarnikova_flux['GM12'] >= 0) & (jarnikova_flux['GM12'].notna())].reindex_like(jarnikova_clim)
jarnikova_flux['SD02'] = pd.Series(jarnikova_flux['SD02'], index=lana['SST'].loc[[12,1,2]].groupby(['latbins','lonbins']).mean().index, name='DMS flux')
#-----------------------------------------------------------------------------
del mat

#%% Compute KDEs for fluxes
def KDE(y):
    """
    A modifed wrapper function pulled from the Pandas source code 
    (https://github.com/pandas-dev/pandas/blob/0.21.x/pandas/plotting/_core.py#L1381-L1430)
    that returns the kernel density estimates of a Pandas Series/sliced DataFrame
    using scipy's gaussian_kde function. It is efficient like the pandas native
    plotting function (because it only fits a subset of only 1000 points from the
    distribution) but it returns the actual values instead of an axes handle.

    Parameters
    ----------
    y : Series or sliced Dataframe
        Input data.

    Returns
    -------
    evals : Series or Dataframe
        col1: Fitted indices (1000 samples between data max/min bounds);
        col2: evaluated kernel density estimates at each indice.

    """
    from scipy.stats import gaussian_kde
    y = y.dropna()
    sample_range = np.nanmax(y) - np.nanmin(y)
    ind = np.linspace(np.nanmin(y) - 0.5 * sample_range,
                      np.nanmax(y) + 0.5 * sample_range, 1000)
    kde = gaussian_kde(y.dropna())
    vals = kde.evaluate(ind)
    evals = pd.concat([pd.Series(ind, name='ind'), pd.Series(vals, name='kde')],axis=1)
    return evals

# Function speeds up computation, but its still faster to load up the data
# rather than rerun the function:
if first_run is True:
    # Calculate the KDEs
    lana_kde = KDE(lana_flux['GM12'])
    jarnikova_kde = KDE(jarnikova_flux['GM12'])
    RFR_kde = KDE(RFR_flux['GM12'])
    RFR_kde_3mon = KDE(RFR_flux['GM12'].loc[[12,1,2],:,:])
    ANN_kde = KDE(ANN_flux['GM12'])
    ANN_kde_3mon = KDE(ANN_flux['GM12'].loc[[12,1,2],:,:])
    # Write each to a csv files
    lana_kde.to_csv(write_dir[:-14]+'lana_kde.csv')
    jarnikova_kde.to_csv(write_dir[:-14]+'jarnikova_kde.csv')
    RFR_kde.to_csv(write_dir[:-14]+'RFR_kde.csv')
    RFR_kde_3mon.to_csv(write_dir[:-14]+'RFR_kde_3mon.csv')
    ANN_kde.to_csv(write_dir[:-14]+'ANN_kde.csv')
    ANN_kde_3mon.to_csv(write_dir[:-14]+'ANN_kde_3mon.csv')
else:
    # load up the csv files
    lana_kde = pd.read_csv(write_dir[:-14]+'lana_kde.csv')
    jarnikova_kde = pd.read_csv(write_dir[:-14]+'jarnikova_kde.csv')
    RFR_kde = pd.read_csv(write_dir[:-14]+'RFR_kde.csv')
    RFR_kde_3mon = pd.read_csv(write_dir[:-14]+'RFR_kde_3mon.csv')
    ANN_kde = pd.read_csv(write_dir[:-14]+'ANN_kde.csv')
    ANN_kde_3mon = pd.read_csv(write_dir[:-14]+'ANN_kde_3mon.csv')

#%% Convert fluxes to Tg S
bounds=[max_lat, min_lat, max_lon, min_lon]

def convert_fluxes(fluxes, to_mask, unique_dates, bounds):
    from calendar import monthrange
    max_lat, min_lat, max_lon, min_lon = bounds[0], bounds[1], bounds[2], bounds[3]
    frac_ocean = to_mask.dropna().size/to_mask.size # calculate the fraction of the total bounded area that is ocean 
    A = ((max_lat-min_lat)*111*1000)*((max_lon-min_lon)*111*1000) # total regional area
    A_ocean = A*frac_ocean # fraction of total area covered by ocean
    S_mol_mass = 32.06 # molar mass of sulfur
    # Get unique dates - average across leap years in the climatology
    dates = pd.DataFrame(np.array([list(monthrange(int(i[:4]),int(i[5:]))) for i in unique_dates]))
    dates = dates.set_index(0)
    num_days = dates.groupby(0).mean().sum().values
    return (fluxes*S_mol_mass*A_ocean*num_days)/(1e6*1e12)

print('RFR')
print(f'{convert_fluxes(RFR_flux["GM12"], to_mask, unique_dates, bounds).mean():.1f}',
      '+/-',
      f'{convert_fluxes(RFR_flux["GM12"], to_mask, unique_dates, bounds).std():.1f}')
print(f'{convert_fluxes(RFR_flux["SD02"], to_mask, unique_dates, bounds).mean():.1f}',
      '+/-',
      f'{convert_fluxes(RFR_flux["SD02"], to_mask, unique_dates, bounds).std():.1f}')

print('\nANN')
print(f'{convert_fluxes(ANN_flux["GM12"], to_mask, unique_dates, bounds).mean():.1f}',
      '+/-',
      f'{convert_fluxes(ANN_flux["GM12"], to_mask, unique_dates, bounds).std():.1f}')
print(f'{convert_fluxes(ANN_flux["SD02"], to_mask, unique_dates, bounds).mean():.1f}',
      '+/-',
      f'{convert_fluxes(ANN_flux["SD02"], to_mask, unique_dates, bounds).std():.1f}')

print('\nCombined')
print(f'{np.nanmean([convert_fluxes(RFR_flux["GM12"], to_mask, unique_dates, bounds), convert_fluxes(ANN_flux["GM12"], to_mask, unique_dates, bounds)]):.1f}',
      '+/-',
      f'{np.nanstd([convert_fluxes(RFR_flux["GM12"], to_mask, unique_dates, bounds), convert_fluxes(ANN_flux["GM12"], to_mask, unique_dates, bounds)]):.1f}')
print(f'{np.nanmean([convert_fluxes(RFR_flux["SD02"], to_mask, unique_dates, bounds), convert_fluxes(ANN_flux["SD02"], to_mask, unique_dates, bounds)]):.1f}',
      '+/-',
      f'{np.nanstd([convert_fluxes(RFR_flux["SD02"], to_mask, unique_dates, bounds), convert_fluxes(ANN_flux["SD02"], to_mask, unique_dates, bounds)]):.1f}')

print('\nCumulative Combined')
print(f'{np.nanmean([convert_fluxes(RFR_flux["GM12"], to_mask, unique_dates, bounds).groupby(["latbins","lonbins"]).sum(), convert_fluxes(ANN_flux["GM12"], to_mask, unique_dates, bounds).groupby(["latbins","lonbins"]).sum()]):.1f}',
      '+/-',
      f'{np.std([convert_fluxes(RFR_flux["GM12"], to_mask, unique_dates, bounds).groupby(["latbins","lonbins"]).sum(), convert_fluxes(ANN_flux["GM12"], to_mask, unique_dates, bounds).groupby(["latbins","lonbins"]).sum()]):.1f}')
print(f'{np.nanmean([convert_fluxes(RFR_flux["SD02"], to_mask, unique_dates, bounds).groupby(["latbins","lonbins"]).sum(), convert_fluxes(ANN_flux["SD02"], to_mask, unique_dates, bounds).groupby(["latbins","lonbins"]).sum()]):.1f}',
      '+/-',
      f'{np.std([convert_fluxes(RFR_flux["SD02"], to_mask, unique_dates, bounds).groupby(["latbins","lonbins"]).sum(), convert_fluxes(ANN_flux["SD02"], to_mask, unique_dates, bounds).groupby(["latbins","lonbins"]).sum()]):.1f}')

print('\nPropagated uncertainity between cumulative combined flux')
print(np.nanmean([convert_fluxes(RFR_flux["GM12"], to_mask, unique_dates, bounds).groupby(["latbins","lonbins"]).sum(),
            convert_fluxes(ANN_flux["GM12"], to_mask, unique_dates, bounds).groupby(["latbins","lonbins"]).sum(),
            convert_fluxes(RFR_flux["SD02"], to_mask, unique_dates, bounds).groupby(["latbins","lonbins"]).sum(),
            convert_fluxes(ANN_flux["SD02"], to_mask, unique_dates, bounds).groupby(["latbins","lonbins"]).sum()]))
print('+/-')
print(np.sqrt(np.sum([np.nanstd(convert_fluxes(RFR_flux["GM12"], to_mask, unique_dates, bounds).groupby(["latbins","lonbins"]).sum())**2,
                np.nanstd(convert_fluxes(ANN_flux["GM12"], to_mask, unique_dates, bounds).groupby(["latbins","lonbins"]).sum())**2,
                np.nanstd(convert_fluxes(RFR_flux["SD02"], to_mask, unique_dates, bounds).groupby(["latbins","lonbins"]).sum())**2,
                np.nanstd(convert_fluxes(ANN_flux["SD02"], to_mask, unique_dates, bounds).groupby(["latbins","lonbins"]).sum())**2]))*0.5)

#%% MLR, LM & LIT models - run to validate models

#-----------------------------------------------------------------------------
#### Linear regression models (run 1 for each variable)
LM_preds = np.empty([y_test.shape[0],np.shape(X_test.columns)[0]])
LM_R2 = np.empty([np.shape(X_test.columns)[0],1])
LM_coef = np.empty([np.shape(X_test.columns)[0],1])
LM_RMSE = np.empty([np.shape(X_test.columns)[0],1])
lm = linear_model.LinearRegression()

for i, var_ in enumerate(X_test.columns.values):
    LM_model = lm.fit(X_test.loc[:,[var_]],np.sinh(y_test))
    ypred_LM = lm.predict(X_test.loc[:,[var_]])
    LM_preds[:,i] = ypred_LM
    LM_R2[i,:] = lm.score(X_test.loc[:,[var_]],np.sinh(y_test))
    LM_coef[i,:] = lm.coef_
    LM_RMSE[i,:] = np.sqrt(metrics.mean_squared_error(np.sinh(y_test), ypred_LM))

#-----------------------------------------------------------------------------
#### Calculate stds, pearson correlations for linear regression models:
LM_stds = np.std(np.arcsinh(LM_preds), axis=0)

LM_corrcoefs = np.empty([LM_preds.shape[1]])
for i in range(LM_preds.shape[1]):
    rs = pearsonr(np.arcsinh(LM_preds[:,i]),y_test)
    LM_corrcoefs[i] = rs[0]
    
R2_LM = np.empty([LM_preds.shape[1]])
for i in range(LM_preds.shape[1]):
    R2_LM[i] = r2_score(y_test, np.arcsinh(LM_preds[:,i]))

print()
print('Linear Regression Results: ')
d = {'Variable':[x for x in X_test.columns.values],'Coefs':LM_coef[:,0],'R2':LM_R2[:,0],'RMSE':LM_RMSE[:,0]}
LM_results = pd.DataFrame(data=d).sort_values('RMSE')
print(LM_results)
print()
#-----------------------------------------------------------------------------
#### MLR
lm_MLR = linear_model.LinearRegression()
MLR_model = lm_MLR.fit(X_train,np.sinh(y_train))
ypred_MLR_train = lm_MLR.predict(X_train) #y predicted by MLR

lm_MLR = linear_model.LinearRegression()
MLR_model = lm_MLR.fit(X_test,np.sinh(y_test))
ypred_MLR = lm_MLR.predict(X_test) #y predicted by MLR
intercept_MLR = lm_MLR.intercept_ #intercept predicted by MLR
coef_MLR = lm_MLR.coef_ #regression coefficients in MLR model
R2_MLR = lm_MLR.score(X_test,np.sinh(y_test)) #R-squared value from MLR model
RMSE_MLR = np.sqrt(metrics.mean_squared_error(np.sinh(y_test), ypred_MLR))
#-----------------------------------------------------------------------------
#### Calculate stds, pearson correlations for multiple linear regression model:
MLR_stds = np.std(np.arcsinh(ypred_MLR))

MLR_corrcoefs = pearsonr(np.arcsinh(ypred_MLR), y_test)[0]

print('MLR results:')
print('a0 (intercept) = ' + str(intercept_MLR)[:5])
for i, val in enumerate(coef_MLR):
    print('a%.0f = %.3f' % (i,val)+' ('+X.columns.values[i]+')')
print('')
print('R^2 = ' + str(R2_MLR)[:4])
print('RMSE = ' + str(RMSE_MLR)[:4])
print('')

#-----------------------------------------------------------------------------
# literature algorithms
# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
# First descale X values:
X_test_orig = pd.DataFrame(scaler.inverse_transform(X_test), index=X_test.index, columns=X_test.columns)
#-----------------------------------------------------------------------------
#### SD02 - Simo & Dachs (2002)

# First run model with global coefs from paper:
global_coefs = np.array([5.7, 55.8, 0.6])
def SD02_model(X, a,b,c):
    coefs = np.array([a,b,c])
    # Chl = X.loc[:,['Chlorophyll a']].values
    Chl = X.loc[:,['chl']].values
    MLD = X.loc[:,['MLD']].values
    Chl_MLD = Chl/MLD
    SD02 = np.empty([Chl.shape[0],Chl.shape[1]])
    for i, val in enumerate(Chl_MLD):
        if val < 0.02:
            SD02[i,0] = -np.log(MLD[i])+coefs[0]
        elif val >= 0.02:
            SD02[i,0] = coefs[1]*(Chl_MLD[i])+coefs[2]
    SD02 = SD02[:,0]
    return SD02
SD02 = SD02_model(X_test_orig, global_coefs[0], global_coefs[1], global_coefs[2])

# Now regionally optimize using least squares:
w, _ = scipy.optimize.curve_fit(SD02_model, X_test_orig, np.sinh(y_test), p0=global_coefs)
SD02_ls_optimized = SD02_model(X_test_orig, w[0], w[1], w[2])

# Now transform to compare:
SD02 = np.arcsinh(SD02)
SD02_ls_optimized = np.arcsinh(SD02_ls_optimized)

# Calculate correlation coefficients, R2, and SDs
SD02_stds = np.std(SD02, axis=0)
SD02_corrcoefs = pearsonr(SD02.flatten(), y_test)[0]
R2_SD02 = r2_score(y_test, SD02.flatten())
SD02_ls_optimized_stds = np.std(SD02_ls_optimized, axis=0)
SD02_ls_optimized_corrcoefs = pearsonr(SD02_ls_optimized.flatten(), y_test)[0]
R2_SD02_ls_optimized = r2_score(y_test, SD02_ls_optimized.flatten())

#-----------------------------------------------------------------------------
#### VS07 - Vallina & Simo (2007)

# First run model with global coefs from paper:
global_coefs = np.array([0.492,0.019])
def VS07_model(X, a, b):
    coefs = np.array([a,b])
    PAR = X.loc[:,['PAR']].values
    Kd = vars_interp['Kd'].reindex_like(X).values
    MLD = X.loc[:,['MLD']].values
    z = MLD # surface depth in m
    SRD = (PAR/(Kd*MLD))*(1-np.exp(-Kd*z))
    VS07 = coefs[0]+(coefs[1]*SRD)
    VS07 = VS07[:,0]
    return VS07
VS07 = VS07_model(X_test_orig, global_coefs[0], global_coefs[1])

# Now regionally optimize using least squares:
w, _ = scipy.optimize.curve_fit(VS07_model, X_test_orig, np.sinh(y_test), p0=global_coefs)
VS07_ls_optimized = VS07_model(X_test_orig, w[0], w[1])

# Now transform to compare:
VS07 = np.arcsinh(VS07)
VS07_ls_optimized = np.arcsinh(VS07_ls_optimized)

# Calculate correlation coefficients, R2, and SDs
VS07_stds = np.std(VS07, axis=0)
VS07_corrcoefs = pearsonr(VS07.flatten(), y_test)[0]
R2_VS07 = r2_score(y_test, VS07.flatten())
VS07_ls_optimized_stds = np.std(VS07_ls_optimized, axis=0)
VS07_ls_optimized_corrcoefs = pearsonr(VS07_ls_optimized.flatten(), y_test)[0]
R2_VS07_ls_optimized = r2_score(y_test, VS07_ls_optimized.flatten())
#-----------------------------------------------------------------------------
#### G18 - Gali et al. (2018)

# First run model with global coefs from paper:
global_coefs = np.array([1.237,0.578,0.0180])
def G18_model(X,a,b,c):
    coefs = np.array([a,b,c])
    Kd = vars_interp['Kd'].reindex_like(X).values.reshape(-1,1)
    MLD = X.loc[:,['MLD']].values
    Chl = X.loc[:,['chl']].values
    SST = X.loc[:,['SST']].values
    PAR = X.loc[:,['PAR']].values
    
    Z_eu = 4.6/Kd # euphotic layer depth
    Z_eu_MLD = Z_eu/MLD
    DMSPt = np.empty([MLD.shape[0], MLD.shape[1]])
    for i,val in enumerate(Z_eu_MLD):
        if val >= 1:
            DMSPt[i,0] = (1.70+(1.14*np.log10(Chl[i]))\
                              +(0.44*np.log10(Chl[i]**2))\
                                  +(0.063*SST[i])-(0.0024*(SST[i]**2)))
        elif val < 1:
            DMSPt[i,0] = (1.74+(0.81*np.log10(Chl[i]))+(0.60*np.log10(Z_eu_MLD[i])))
    G18 = -coefs[0]+(coefs[1]*DMSPt)+(coefs[2]*PAR)
    G18 = 10**(G18[:,0])
    return G18
G18 = G18_model(X_test_orig, global_coefs[0],global_coefs[1],global_coefs[2])

#### Now regionally optimize using least squares:
w, _ = scipy.optimize.curve_fit(G18_model, X_test_orig, np.sinh(y_test), p0=global_coefs)
G18_ls_optimized = G18_model(X_test_orig, w[0], w[1], w[2])

#### Now transform to compare:
G18 = np.arcsinh(G18)
G18_ls_optimized = np.arcsinh(G18_ls_optimized)

#### Calculate correlation coefficients, R2, and SDs
G18_stds = np.std(G18, axis=0)
G18_corrcoefs = pearsonr(G18.flatten(), y_test)[0]
R2_G18 = r2_score(y_test, G18.flatten())
G18_ls_optimized_stds = np.std(G18_ls_optimized, axis=0)
G18_ls_optimized_corrcoefs = pearsonr(G18_ls_optimized.flatten(), y_test)[0]
R2_G18_ls_optimized = r2_score(y_test, G18_ls_optimized.flatten())

#### Print results
table=[['SD02',R2_SD02],['SD02 LS',R2_SD02_ls_optimized],['VS07',R2_VS07],['VS07 LS',R2_VS07_ls_optimized],['G18',R2_G18],['G18 R2',R2_G18_ls_optimized]]
print(tabulate(table, headers=['Model','R2']))

#%% ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

#%% Plot Taylor Diagram
#-----------------------------------------------------------------------------
# Calculate the std of DMS data
stdrefs = y_test.std()
#-----------------------------------------------------------------------------
# Plot Taylor Diagram:
fig = plt.figure(figsize=(12,12))
font={'family':'sans-serif',
      'weight':'normal',
      'size':'24'} 
plt.rc('font', **font) # sets the specified font formatting globally

dia = TaylorDiagram(stdrefs, fig=fig, rect=111, srange=(0,1.31),
                    label='Reference', extend=False, normalize=True)

contours = dia.add_contours(levels=[0,0.25,0.5,0.75,1,1.25,1.5], colors='r')  # 5 levels in grey
plt.clabel(contours, inline=1, fontsize=20, fmt='%.2f')
dia.add_text(0.87,0.14,s='RMSE', fontsize=16, color='r', rotation=50)

# Add RFR values
dia.add_sample(RFR_stds,
               RFR_corrcoefs,
               marker='o',
               ms=10,
               ls='',
               mfc='b',
               mec='k',
               label='RFR',
               normalize=True,
               zorder=2)

# Add ANN values
dia.add_sample(ANN_stds,
               ANN_corrcoefs,
               marker='o', ms=10, ls='',
               mfc='r', mec='k',
               label='ANN',
               normalize=True,
               zorder=1)

# Add MLR values
dia.add_sample(MLR_stds,
               MLR_corrcoefs,
               marker='+',
               ms=20,
               ls='',
               mew='4',
               mfc='k',
               mec='deepskyblue',
               label="MLR",
               normalize=True,
               zorder=10)

# Add Optimized SD02 values
dia.add_sample(SD02_ls_optimized_stds,
               SD02_ls_optimized_corrcoefs,
               marker='s',
               ms=15,
               ls='',
               mew='1',
               mfc='yellow',
               mec='k',
               label="SD02 (LS optimized)",
               normalize=True,
               zorder=11)

# Add Optimized VS07 values
dia.add_sample(VS07_ls_optimized_stds,
               VS07_ls_optimized_corrcoefs,
               marker='D',
               ms=15,
               ls='',
               mew='1',
               mfc='cyan',
               mec='k',
               label="VS07 (LS optimized)",
               normalize=True,
               zorder=10)

# Add Optimized G18 values
dia.add_sample(G18_ls_optimized_stds,
               G18_ls_optimized_corrcoefs,
               marker='^',
               ms=15,
               ls='',
               mew='1',
               mfc='darkorange',
               mec='k',
               label="G18 (LS optimized)",
               normalize=True,
               zorder=9)

# Add SD02 values
dia.add_sample(SD02_stds,
               SD02_ls_optimized_corrcoefs,
               marker='s',
               ms=15,
               ls='',
               mew='1',
               mfc='r',
               mec='k',
               label="SD02",
               normalize=True,
               zorder=11)
# Add VS07 values
dia.add_sample(VS07_stds,
               VS07_ls_optimized_corrcoefs,
               marker='D',
               ms=15,
               ls='',
               mew='1',
               mfc='r',
               mec='k',
               label="VS07",
               normalize=True,
               zorder=10)
# Add G18 values
dia.add_sample(G18_stds,
               G18_corrcoefs,
               marker='^',
               ms=15,
               ls='',
               mew='1',
               mfc='r',
               mec='k',
               label="G18",
               normalize=True,
               zorder=9)

dia._ax.axis[:].major_ticks.set_tick_out(True)
dia.add_grid(lw=1, ls='--')
fig.legend(dia.samplePoints,
                [ p.get_label() for p in dia.samplePoints ],
                numpoints=1, bbox_to_anchor=(1.2, 0.9), prop=dict(size='small'), loc='upper right', facecolor='none')

# plt.tight_layout()

fig.savefig(save_to_path+str('Taylor.png'), dpi=500, bbox_inches='tight')
#%% Plot MLR vs. RFR vs. ANN
fig = plt.figure(figsize=(24, 6))
# fig = plt.figure(figsize=(9,18)) # if stacking vertically
font={'family':'DejaVu Sans',
      'weight':'normal',
      'size':'24'} 
plt.rc('font', **font) # sets the specified font formatting globally
ax3=fig.add_subplot(131)
ax4=fig.add_subplot(133)
ax5=fig.add_subplot(132)
#-----------------------------------------------------------------------------
# Plot MLR Fit
ax3.scatter(y_train,np.arcsinh(ypred_MLR_train),s=10,c='k',marker='+',label='Training')
ax3.scatter(y_test,np.arcsinh(ypred_MLR),s=10,c='deepskyblue',marker='+',label='Testing (R${^2}$ = ' + str(round(R2_MLR,2))+')')
l1 = np.min(ax3.get_xlim())
l2 = np.max(ax3.get_xlim())
ax3.plot([l1,l2], [l1,l2], ls="--", c=".3", zorder=0)
ax3.set_xlim([0, RFR_model.predict(scaler.transform(X)).max()])
ax3.set_xlabel(r'arcsinh(DMS$_{\rmmeasured}$)')
ax3.set_ylabel(r'arcsinh(DMS$_{\rmmodel}$)')
ax3.legend(loc='upper center', markerscale=3, fontsize=20, facecolor='none')
ax3.text(0.79,0.06,'MLR', transform=ax3.transAxes, fontweight='bold',bbox=dict(facecolor='none',edgecolor='k'))
ax3.text(0.03,0.9,'$\mathbf{a}$',fontsize='24',transform=ax3.transAxes)
#-----------------------------------------------------------------------------
# Plot ANN Fit
ax4.scatter(y_train,ANN_y_train_pred,s=10,c='k', label="Training")
ax4.scatter(y_test,ANN_y_test_pred,s=10,c='r', label="Testing (R${^2}$ = "+ str(round(ANN_ensemble_R2,2))+")")
l1 = np.min(ax4.get_xlim())
l2 = np.max(ax4.get_xlim())
ax4.plot([l1,l2], [l1,l2], ls="--", c=".3", zorder=0)
ax4.set_xlim([0, RFR_model.predict(scaler.transform(X)).max()])
ax4.set_xlabel(r'arcsinh(DMS$_{\rmmeasured}$)')
# ax4.set_ylabel(r'arcsinh(DMS$_{\rmmodel}$)')
ax4.legend(loc='upper center', markerscale=3, fontsize=20, facecolor='none')
ax4.text(0.79,0.06,'ANN', transform=ax4.transAxes,fontweight='bold',bbox=dict(facecolor='none',edgecolor='k'))
ax4.text(0.03,0.9,'$\mathbf{c}$',fontsize='24',transform=ax4.transAxes)
#-----------------------------------------------------------------------------
# Plot RFR Fit
ax5.scatter(y_train,RFR_model.predict(X_train),s=10,c='k', label="Training")
ax5.scatter(y_test,RFR_model.predict(X_test),s=10,c='b', label="Testing (R${^2}$ = "+ str(round(RFR_model_R2,2))+")")
l1 = np.min(ax5.get_xlim())
l2 = np.max(ax5.get_xlim())
ax5.plot([l1,l2], [l1,l2], ls="--", c=".3", zorder=0)
ax5.set_xlim([0, RFR_model.predict(scaler.transform(X)) .max()])
ax5.set_xlabel(r'arcsinh(DMS$_{\rmmeasured}$)')
# ax5.set_ylabel(r'arcsinh(DMS$_{\rmmodel}$)')
ax5.legend(loc='upper center', markerscale=3, fontsize=20, facecolor='none')
ax5.text(0.79,0.06,'RFR', transform=ax5.transAxes,fontweight='bold',bbox=dict(facecolor='none',edgecolor='k'))
ax5.text(0.03,0.9,'$\mathbf{b}$',fontsize='24',transform=ax5.transAxes)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
fig.subplots_adjust(hspace=2)
# fig.subplots_adjust(wspace=0.1) # if stacking vertically

# fig.savefig(save_to_path+str('Ensemble_performance.png'), dpi=500, transparent=True, bbox_inches='tight')


#%% Print summary statistics for fluxes
from tabulate import tabulate

param_ = 'GM12'
headers = ['model', 'mean', 'SD', 'min', 'max']
data = [('RFR',np.nanmean(RFR_flux[param_]), np.nanstd(RFR_flux[param_]),np.nanmin(RFR_flux[param_]),np.nanmax(RFR_flux[param_])),
        ('ANN',np.nanmean(ANN_flux[param_]), np.nanstd(ANN_flux[param_]),np.nanmin(ANN_flux[param_]),np.nanmax(ANN_flux[param_])),
        ('L11',np.nanmean(lana_flux[param_]), np.nanstd(lana_flux[param_]),np.nanmin(lana_flux[param_]),np.nanmax(lana_flux[param_])),
        ('JT16',np.nanmean(jarnikova_flux[param_]), np.nanstd(jarnikova_flux[param_]),np.nanmin(jarnikova_flux[param_]),np.nanmax(jarnikova_flux[param_]))]
print(f'\nParmaeterization: {param_}')
print(tabulate(data, headers))

param_ = 'SD02'
headers = ['model', 'mean', 'SD', 'min', 'max']
data = [('RFR',np.nanmean(RFR_flux[param_]), np.nanstd(RFR_flux[param_]),np.nanmin(RFR_flux[param_]),np.nanmax(RFR_flux[param_])),
        ('ANN',np.nanmean(ANN_flux[param_]), np.nanstd(ANN_flux[param_]),np.nanmin(ANN_flux[param_]),np.nanmax(ANN_flux[param_])),
        ('L11',np.nanmean(lana_flux[param_]), np.nanstd(lana_flux[param_]),np.nanmin(lana_flux[param_]),np.nanmax(lana_flux[param_])),
        ('JT16',np.nanmean(jarnikova_flux[param_]), np.nanstd(jarnikova_flux[param_]),np.nanmin(jarnikova_flux[param_]),np.nanmax(jarnikova_flux[param_]))]
print(f'\nParmaeterization: {param_}')
print(tabulate(data, headers))

#%% Plot sea-air fluxes - compare climatologies

label_size = 32

fig = plt.figure(figsize=(24,38))
font={'family':'DejaVu Sans',
      'weight':'normal',
      'size':'24'} 
plt.rc('font', **font) # sets the specified font formatting globally
gs = fig.add_gridspec(10, 2)
# main plots
# Increase resolution of projection - needed to draw polygons accurately
map_proj = ccrs.Orthographic(central_latitude=-90.0, central_longitude=0)
map_proj._threshold /= 100
ax = fig.add_subplot(gs[0:3, 0], projection=map_proj)
ax2 = fig.add_subplot(gs[3:6, 0], projection=map_proj)
ax3 = fig.add_subplot(gs[0:3, 1], projection=map_proj)
ax4 = fig.add_subplot(gs[3:6, 1], projection=map_proj)
# cax = fig.add_subplot(gs[6,0:]) # for colorbar
cax = fig.add_axes([ax2.get_position().x0,
                    ax2.get_position().y0-0.05,
                    (ax2.get_position().x1-ax2.get_position().x0)*2+(ax3.get_position().x0-ax2.get_position().x1),
                    0.02])
ax5 = fig.add_subplot(gs[7:9, 0])
ax6 = fig.add_subplot(gs[9, 0])
ax7 = fig.add_subplot(gs[7:9, 1])
ax8 = fig.add_subplot(gs[9, 1])

vmin=0
vmax=30
#------------------------------------------------------------------------------
# Plot ML fluxes (full climatology)
h, ax = South_1ax_map(ax=ax,
                      data=fluxes_combined.groupby(['latbins','lonbins']).mean().unstack('lonbins'),
                      vmin=vmin,
                      vmax=vmax,
                      cmap=cmocean.cm.haline)
ax.set_title('Mean ML Predictions \nOct-Apr, 20.0 km')
ax.text(-0.05,1,'$\mathbf{a}$',fontsize=label_size,transform=ax.transAxes, zorder=500)
#------------------------------------------------------------------------------
# Plot Lana Climatology
h2, ax2 = South_1ax_map(ax=ax2,
                        data=lana_flux['GM12'].groupby(['latbins','lonbins']).mean().unstack('lonbins'),
                        vmin=vmin,
                        vmax=vmax,
                        cmap=cmocean.cm.haline)
ax2.set_title(f'L11 Climatology \nOct-Apr, {degrees2kilometers(1):.1f} km')
ax2.text(-0.05,1,'$\mathbf{c}$',fontsize=label_size,transform=ax2.transAxes, zorder=500)
#------------------------------------------------------------------------------
# Plot ML fluxes (full climatology)
h, ax3 = South_1ax_map(ax=ax3,
                      data=fluxes_combined.loc[[12,1,2],:,:].groupby(['latbins','lonbins']).mean().unstack('lonbins'),
                      vmin=vmin,
                      vmax=vmax,
                      cmap=cmocean.cm.haline)
ax3.set_title('Mean ML Predictions \nDec-Feb, 20.0 km')
ax3.text(-0.05,1,'$\mathbf{b}$',fontsize=label_size,transform=ax3.transAxes, zorder=500)
#------------------------------------------------------------------------------
# Plot Jarnikova Climatology
h5, ax4 = South_1ax_map(ax=ax4,
                        data=jarnikova_flux['GM12'].unstack('lonbins'),
                        vmin=vmin,
                        vmax=vmax,
                        cmap=cmocean.cm.haline)
cb = plt.colorbar(h, cax=cax, fraction=0.001, extend='max', orientation='horizontal')
cb.set_ticks(np.arange(vmin,vmax+(vmax-vmin)/10,(vmax-vmin)/10))
cb.set_ticklabels(np.arange(vmin,vmax+(vmax-vmin)/10,(vmax-vmin)/10))
cb.set_label(r'DMS flux ($\mathrm{\mu}$mol $\mathrm{m^{-2}}$ $\mathrm{d^{-1}}$)', size=label_size)
ax4.set_title(f'JT16 Climatology \nDec-Feb, {degrees2kilometers(1):.1f} km')
ax4.text(-0.05,1,'$\mathbf{d}$',fontsize=label_size,transform=ax4.transAxes, zorder=500)
#------------------------------------------------------------------------------
# Plot KDEs of flux distributions (ML vs. Lana)
max_kde = []
for i in [lana_kde, RFR_kde, ANN_kde]:
    max_kde.append(i['kde'].max())
max_kde = np.array(max_kde)

ax5.plot(RFR_kde['ind'],RFR_kde['kde']/max_kde.max(),'b-', lw=2, label='RFR')
ax5.plot(ANN_kde['ind'],ANN_kde['kde']/max_kde.max(),'r-', lw=2, label='ANN')
ax5.plot(lana_kde['ind'],lana_kde['kde']/max_kde.max(),c='gray',ls='--', lw=2, label='L11')

ax5.set_xlim(0,30)
ax5.set_ylim(0,1)
ax5.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
ax5.tick_params('both', length=10, width=1, which='major')
ax5.tick_params('both', length=5, width=1, which='minor')
ax5.set_xticks([])
ax5.legend()
ax5.set_ylabel('Probability Density (norm.)')
ax5.text(-0.05,1.05,'$\mathbf{e}$',fontsize=label_size,transform=ax5.transAxes, zorder=500)
#------------------------------------------------------------------------------
# Plot boxplot of fluxes (ML vs. Lana)
labels = ['RFR', 'ANN', 'L11']
bplot1 = ax6.boxplot([RFR_flux['GM12'].dropna(),ANN_flux['GM12'].dropna(),lana_flux['GM12'].dropna()],
                    widths=0.5,
                    vert=False,
                    showfliers=False,
                    patch_artist=True,  # fill with color
                    labels=labels)

# fill with colors
colors = ['blue', 'red', 'gray']
for patch, color in zip(bplot1['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
for patch in bplot1['medians']:
    patch.set_color('black')

ax6.set_xlim([0,30])
ax6.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
ax6.tick_params('both', length=10, width=1, which='major')
ax6.tick_params('both', length=5, width=1, which='minor')
ax6.set_xlabel(r"DMS flux ($\mathrm{\mu}$mol $\mathrm{m^{-2}}$ $\mathrm{d^{-1}}$)")
ax6.text(-0.05,1.05,'$\mathbf{g}$',fontsize=label_size-2,transform=ax6.transAxes, zorder=1000)
#------------------------------------------------------------------------------
# Plot KDEs of flux distributions (ML vs. Jarnikova) - normalized
max_kde = []
for i in [jarnikova_kde, RFR_kde_3mon, ANN_kde_3mon]:
    max_kde.append(i['kde'].max())
max_kde = np.array(max_kde)

ax7.plot(RFR_kde_3mon['ind'],RFR_kde_3mon['kde']/max_kde.max(),'b-', lw=2, label='RFR')
ax7.plot(ANN_kde_3mon['ind'],ANN_kde_3mon['kde']/max_kde.max(),'r-', lw=2, label='ANN')
ax7.plot(jarnikova_kde['ind'],jarnikova_kde['kde']/max_kde.max(),c='darkorange',ls='--', lw=2, label='JT16')

ax7.set_xlim(0,30)
ax7.set_ylim(0,1)
ax7.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
ax7.tick_params('both', length=10, width=1, which='major')
ax7.tick_params('both', length=5, width=1, which='minor')
ax7.set_xticks([])
ax7.legend()
ax7.text(-0.05,1.05,'$\mathbf{f}$',fontsize=label_size,transform=ax7.transAxes, zorder=500)
#------------------------------------------------------------------------------
# Plot boxplot of fluxes (ML vs. Jarnikova)
labels = ['RFR', 'ANN', 'JT16']
bplot1 = ax8.boxplot([RFR_flux['GM12'].loc[[12,1,2],:,:].dropna(),ANN_flux['GM12'].loc[[12,1,2],:,:].dropna(),jarnikova_flux['GM12'].dropna()],
                    widths=0.5,
                    vert=False,
                    showfliers=False,
                    patch_artist=True,  # fill with color
                    labels=labels)

# fill with colors
colors = ['blue', 'red', 'darkorange']
for patch, color in zip(bplot1['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
for patch in bplot1['medians']:
    patch.set_color('black')

ax8.set_xlim([0,30])
ax8.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
ax8.tick_params('both', length=10, width=1, which='major')
ax8.tick_params('both', length=5, width=1, which='minor')
ax8.set_xlabel(r"DMS flux ($\mathrm{\mu}$mol $\mathrm{m^{-2}}$ $\mathrm{d^{-1}}$)")
ax8.text(-0.05,1.05,'$\mathbf{h}$',fontsize=label_size-2,transform=ax8.transAxes, zorder=1000)

fig.subplots_adjust(hspace=0.5)

#%% Map predictions, observations, and deviance
label_size = 32
sizing = 24
#------------------------------------------------------------------------------
# set color scale range
vmin = 0
vmax = 10
# set deviance scale range
dev_vmin = -5
dev_vmax = 5
# colorbar step size
dev_step = (dev_vmax-dev_vmin)/10
step = 0.25
#------------------------------------------------------------------------------

# Map the averages
# fig = plt.figure(figsize=(36,18))
fig = plt.figure(figsize=(24,24))
font={'family':'DejaVu Sans',
      'weight':'normal',
      'size':'22'} 
plt.rc('font', **font) # sets the specified font formatting globally
ax = fig.add_subplot(2,2,1, projection=ccrs.Orthographic(central_longitude=0, central_latitude=-90))
ax2 = fig.add_subplot(2,2,2, projection=ccrs.Orthographic(central_longitude=0, central_latitude=-90))
ax3 = fig.add_subplot(2,2,3, projection=ccrs.Orthographic(central_longitude=0, central_latitude=-90))
ax4 = fig.add_subplot(2,2,4, projection=ccrs.Orthographic(central_longitude=0, central_latitude=-90))

#-----------------------------------------------------------------------------
#### Map RFR
h, ax = South_1ax_map(ax=ax,
                      data=RFR_y_pred_mean.unstack('lonbins'),
                      plottype='mesh',
                      cmap=cmocean.cm.haline,
                      vmin=vmin,
                      vmax=vmax,
                      extend="max",
                      )
newcmap = cmocean.tools.crop_by_percent(cmocean.cm.turbid, 50, which='max', N=None)
cs = newcmap(np.linspace(0,1,4))
hs = []
for i,key in enumerate(list(fronts.keys())[:-1]):
   h = ax.scatter(x=fronts[key].index.get_level_values('lonbins').values,
                y=fronts[key].index.get_level_values('latbins').values,
                s=fronts[key].values,
                color=cs[i],
                transform=ccrs.PlateCarree())
   hs.append(h)
ax.legend(handles=hs, labels=fronts.keys(), loc='center', bbox_to_anchor=(0.9,-0.225,0.5,0.5), markerscale=15, prop={'size': 30})
for ha in ax.legend_.legendHandles:
    ha.set_edgecolor("k")
ax.set_title('RFR')
ax.text(-0.05,1,'$\mathbf{a}$',fontsize=label_size,transform=ax.transAxes, zorder=500)
#-----------------------------------------------------------------------------
#### Map ANN
h2, ax2 = South_1ax_map(ax=ax2,
                        data=ANN_y_pred_mean.unstack('lonbins'),
                        plottype='mesh',
                        cmap=cmocean.cm.haline,
                        vmin=vmin,
                        vmax=vmax,
                        extend="max",
                        )
newcmap = cmocean.tools.crop_by_percent(cmocean.cm.turbid, 50, which='max', N=None)
cs = newcmap(np.linspace(0,1,4))
hs = []
for i,key in enumerate(list(fronts.keys())[:-1]):
   h = ax2.scatter(x=fronts[key].index.get_level_values('lonbins').values,
                y=fronts[key].index.get_level_values('latbins').values,
                s=fronts[key].values,
                color=cs[i],
                transform=ccrs.PlateCarree())
   hs.append(h)

ax2.set_title('ANN')
ax2.text(-0.05,1,'$\mathbf{b}$',fontsize=label_size,transform=ax2.transAxes, zorder=500)
#-----------------------------------------------------------------------------
#### Map deviance
norm = mpl.colors.TwoSlopeNorm(vmin=dev_vmin, vcenter=0, vmax=dev_vmax) # scales to accentuate depth colors, and diverge at 0
h3, ax3 = South_1ax_map(ax=ax3,
                        data=(RFR_y_pred_mean-ANN_y_pred_mean).unstack('lonbins'),
                        plottype='mesh',
                        cmap='RdBu',
                        norm=norm,
                        vmin=dev_vmin,
                        vmax=dev_vmax,
                        extend="both",
                        )
ax3.set_title('Deviation')
# add colorbar
divider = make_axes_locatable(ax3)
ax_cb = divider.new_vertical(size="5%", pad=0.2, axes_class=plt.Axes, pack_start=True)
fig.add_axes(ax_cb)
cb = plt.colorbar(h3, cax=ax_cb, orientation="horizontal")
cb.ax.tick_params(labelsize=sizing)
cb.set_label('Model Deviation (RFR-ANN, nM)', fontsize=sizing)
cb.set_ticks(np.arange(dev_vmin, dev_vmax+dev_step, dev_step))
cb.set_ticklabels(np.arange(dev_vmin, dev_vmax+dev_step, dev_step).astype(int))
ax3.text(-0.05,1,'$\mathbf{c}$',fontsize=label_size,transform=ax3.transAxes, zorder=500)
#-----------------------------------------------------------------------------
#### Map Observations
h4,ax4 = South_1ax_map(ax=ax4, 
                        data=np.sinh(y).groupby(['latbins','lonbins']).mean().unstack('lonbins'),
                        s=5,
                        plottype='scatter',
                        cmap=cmocean.cm.haline,
                        vmin=vmin,
                        vmax=vmax,
                        )
ax4.set_title('Obs.')
# add colorbar
divider = make_axes_locatable(ax4)
ax_cb = divider.new_vertical(size="5%", pad=0.2, axes_class=plt.Axes, pack_start=True)
fig.add_axes(ax_cb)
cb = plt.colorbar(h4, cax=ax_cb, orientation="horizontal", extend='max')
cb.ax.tick_params(labelsize=sizing)
cb.set_label(r'DMS (nM)', fontsize=sizing)
cb.set_ticks(np.arange(vmin, vmax+2, 2))
cb.set_ticklabels(np.arange(vmin, vmax+2, 2))
ax4.text(-0.05,1,'$\mathbf{d}$',fontsize=label_size,transform=ax4.transAxes, zorder=500)

#%% Wind speed vs DMS
label_size = 32

fig = plt.figure(figsize=(34,24))
font={'family':'DejaVu Sans',
      'weight':'normal',
      'size':'22'} 
plt.rc('font', **font) # sets the specified font formatting globally
gs = fig.add_gridspec(2, 4)
# main plots
ax = fig.add_subplot(gs[0:1,0:2], projection=ccrs.Orthographic(central_longitude=0, central_latitude=-90))
ax2 = fig.add_subplot(gs[1:2,0:2], projection=ccrs.Orthographic(central_longitude=0, central_latitude=-90))
ax3 = fig.add_subplot(gs[0:2,2:])
#------------------------------------------------------------------------------
#### Map DMS
vmin=0
vmax=10
h, ax = South_1ax_map(ax=ax,
                      data=models_combined.groupby(['latbins','lonbins']).mean().unstack('lonbins'),
                      plottype='mesh',
                      # levels=np.linspace(vmin, vmax, 100),
                      vmin=vmin,
                      vmax=vmax,
                      extend='max',
                      cmap=cmocean.cm.haline,
                      )
h2 = ax.contour(X_full.loc[:,'MLD'].groupby(['latbins','lonbins']).mean().unstack('lonbins').columns.values,
           X_full.loc[:,'MLD'].groupby(['latbins','lonbins']).mean().unstack('lonbins').index.values,
           X_full.loc[:,'MLD'].groupby(['latbins','lonbins']).mean().unstack('lonbins').values,
           levels=[60],
           colors='w',
           transform=ccrs.PlateCarree())
ax.clabel(h2)
divider = make_axes_locatable(ax)
ax_cb = divider.new_vertical(size="5%", pad=0.2, axes_class=plt.Axes, pack_start=True)
fig.add_axes(ax_cb)
cb = plt.colorbar(h, cax=ax_cb, orientation="horizontal", extend='max')
cb.set_ticks(np.arange(vmin,vmax+(vmax-vmin)/10,(vmax-vmin)/10))
cb.set_ticklabels(np.arange(vmin,vmax+(vmax-vmin)/10,(vmax-vmin)/10))
cb.set_label(r'DMS$_{\rmmodel}$ (nM)', size=22)

ax.text(-0.05,1,'$\mathbf{a}$',fontsize=label_size,transform=ax.transAxes, zorder=500)

#------------------------------------------------------------------------------
#### Map Winds
wind_vmin = X_full.loc[:,'wind'].groupby(['latbins','lonbins']).mean().min()
wind_vmax = X_full.loc[:,'wind'].groupby(['latbins','lonbins']).mean().max()
h3, ax2 = South_1ax_map(ax=ax2,
                      data=X_full.loc[:,'wind'].groupby(['latbins','lonbins']).mean().unstack('lonbins'),
                       plottype='mesh',
                      # levels=np.linspace(wind_vmin, wind_vmax, 100),
                      vmin=wind_vmin,
                      vmax=wind_vmax,
                      cmap='Spectral_r',
                      )
h4 = ax2.contour(X_full.loc[:,'MLD'].groupby(['latbins','lonbins']).mean().unstack('lonbins').columns.values,
           X_full.loc[:,'MLD'].groupby(['latbins','lonbins']).mean().unstack('lonbins').index.values,
           X_full.loc[:,'MLD'].groupby(['latbins','lonbins']).mean().unstack('lonbins').values,
           levels=[60],
           colors='k',
           transform=ccrs.PlateCarree())
ax.clabel(h4)
divider = make_axes_locatable(ax2)
ax_cb = divider.new_vertical(size="5%", pad=0.2, axes_class=plt.Axes, pack_start=True)
fig.add_axes(ax_cb)
cb = plt.colorbar(h3, cax=ax_cb, orientation="horizontal")
cb.set_ticks(np.round(np.arange(2,wind_vmax+1,2),0))
cb.set_ticklabels(np.round(np.arange(2,wind_vmax+1,2),0))
cb.set_label(r'Wind Speed (m $s^{-1}$)', size=22)

ax2.text(-0.05,1,'$\mathbf{b}$',fontsize=label_size,transform=ax2.transAxes, zorder=500)
#------------------------------------------------------------------------------
#### Bivariate wind speed vs DMS

# fit a linear model to the time-averaged data
lm = linear_model.LinearRegression().fit(X_full.loc[:,['wind']].groupby(['latbins','lonbins']).mean(),
                  np.log10(models_combined).groupby(['latbins','lonbins']).mean().to_frame())
wind_r2 = lm.score(X_full.loc[:,['wind']].groupby(['latbins','lonbins']).mean(),
                  np.log10(models_combined).groupby(['latbins','lonbins']).mean().to_frame())
wind_preds = lm.predict(X_full.loc[:,['wind']].groupby(['latbins','lonbins']).mean())

# Plot the data
cbaxes = inset_axes(ax3, width="30%", height="3%", loc='lower left',
                    bbox_to_anchor=(0.03, 0.06, 0.99, 0.9), bbox_transform=ax3.transAxes)
ax3.scatter(X_full.loc[:,'wind'].groupby(['latbins','lonbins']).mean(),
           np.log10(models_combined.groupby(['latbins','lonbins']).mean()),
          # models_combined.groupby(['latbins','lonbins']).mean(),
          s=2,
          color='k',
          marker='.')
h = sns.histplot(x=X_full.loc[:,'wind'].groupby(['latbins','lonbins']).mean(),
              y=np.log10(models_combined.groupby(['latbins','lonbins']).mean()),
             stat='density',
             bins=100, pthresh=.1, cmap="mako", ax=ax3, cbar=True, cbar_ax=cbaxes, cbar_kws={'orientation':'horizontal'}, zorder=2)

# set name and fontsizes of colorbar
h.figure.axes[-1].set_xlabel('Counts', size=22)
h.figure.axes[-1].tick_params(labelsize=22)

ax3.plot(X_full.loc[:,['wind']].groupby(['latbins','lonbins']).mean().values,
          wind_preds,
          'r-',
          zorder=3)
ax3.set_ylabel(r'log$_{10}$(DMS$_{\rmmodel}$)')
ax3.set_xlabel('Wind Speed (m s$^{-1}$)')
ax3.set_xlim(X_full.loc[:,'wind'].groupby(['latbins','lonbins']).mean().min(),
             X_full.loc[:,'wind'].groupby(['latbins','lonbins']).mean().max()+2)
ax3.set_ylim(np.log10(1),np.log10(models_combined.groupby(['latbins','lonbins']).mean().max()))

# Format ticks
ax3.tick_params('both', length=10, width=1, which='major')
ax3.tick_params('both', length=8, width=1, which='minor')
ax3.set_xscale('linear')
ax3.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())

ax3.text(-0.05,1,'$\mathbf{c}$',fontsize=label_size,transform=ax3.transAxes, zorder=500)
#------------------------------------------------------------------------------
#### Miniplot of winds vs. DMS observations
# Define mini axes to plot in
left, bottom, width, height = [0.74, 0.66, 0.15, 0.2]
ax4 = fig.add_axes([left, bottom, width, height])
# Bin data
binned = pd.concat([X.loc[:,'wind'],np.log10(np.sinh(y))], axis=1)
wind_binned = binned.groupby(pd.cut(binned['wind'], np.arange(4,12,1))).mean()
wind_binned_std = binned.groupby(pd.cut(binned['wind'], np.arange(4,12,1))).std()

# Plot scatter of observations
ax4.errorbar(wind_binned['wind'],
             wind_binned['DMS'],
             yerr=wind_binned_std['DMS'],
             capsize=3,
             elinewidth=2,
             ecolor='k',
             ls='None',
             marker='.',
             mfc='k',
             mec='k',
             ms=15)
# Plot the binned data
ax4.scatter(X.loc[:,'wind'].groupby(['latbins','lonbins']).mean(),
          np.log10(np.sinh(y).groupby(['latbins','lonbins']).mean()),
          s=2,
          color='gray',
          marker='.')

# Fit linear model, plot line of best fit
lm = linear_model.LinearRegression().fit(wind_binned[['wind']],wind_binned[['DMS']])
wind_r22 = lm.score(wind_binned[['wind']],wind_binned[['DMS']])
ax4.plot(binned['wind'],
         lm.predict(binned[['wind']]),
         'r-',
         zorder=3)

# Format axis labels and ticks
ax4.set_ylabel(r'log$_{10}$(DMS$_{\rmobs}$)', fontsize=18)
ax4.set_xlabel('Wind Speed (m s$^{-1}$)', fontsize=18)
ax4.set_xlim(X.loc[:,'wind'].groupby(['latbins','lonbins']).mean().min(),
             X.loc[:,'wind'].groupby(['latbins','lonbins']).mean().max())
ax4.set_ylim(np.log10(0.1),np.log10(np.sinh(y).groupby(['latbins','lonbins']).mean().max()))
ax4.tick_params('both', length=10, width=1, which='major')
ax4.tick_params('both', length=8, width=1, which='minor')
ax4.set_xlim(2.5,12.5)
ax4.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())

print(wind_r2)
print(wind_r22)

# fig.subplots_adjust(wspace=0.1)

#%% Map DMS, ice, Si*
label_size = 32

fig = plt.figure(figsize=(24,12))
font={'family':'DejaVu Sans',
      'weight':'normal',
      'size':'22'} 
plt.rc('font', **font) # sets the specified font formatting globally
ax = fig.add_subplot(1,2,1, projection=ccrs.Orthographic(central_longitude=0, central_latitude=-90))
ax2 = fig.add_subplot(1,2,2, projection=ccrs.Orthographic(central_longitude=0, central_latitude=-90))

vmin=0
vmax=10
front_color = 'magenta'
#------------------------------------------------------------------------------
#### Mean DMS
h, ax = South_1ax_map(ax=ax,
                      data=models_combined.groupby(['latbins','lonbins']).mean().unstack('lonbins'),
                      plottype='mesh',
                      # levels=np.linspace(vmin, vmax, 100),
                      vmin=vmin,
                      vmax=vmax,
                      extend='max',
                      cmap=cmocean.cm.haline,
                      )
h2 = ax.contour(X_full_plus.loc[:,'Si_star'].groupby(['latbins','lonbins']).mean().unstack('lonbins').columns.values,
                  X_full_plus.loc[:,'Si_star'].groupby(['latbins','lonbins']).mean().unstack('lonbins').index.values,
                  X_full_plus.loc[:,'Si_star'].groupby(['latbins','lonbins']).mean().unstack('lonbins').values,
                  levels=[-10,0,20,40,50],
                  colors='w',
                  # linewidths=1,
                  transform=ccrs.PlateCarree())
ax.clabel(h2, fontsize=20)

divider = make_axes_locatable(ax)
ax_cb = divider.new_vertical(size="5%", pad=0.2, axes_class=plt.Axes, pack_start=True)
fig.add_axes(ax_cb)
cb = plt.colorbar(h, cax=ax_cb, orientation="horizontal", extend='max')
cb.set_ticks(np.arange(vmin,vmax+(vmax-vmin)/10,(vmax-vmin)/10))
cb.set_ticklabels(np.arange(vmin,vmax+(vmax-vmin)/10,(vmax-vmin)/10))
cb.set_label(r'DMS$_{\rmmodel}$ (nM)', size=22)

ax.text(-0.05,1,'$\mathbf{a}$',fontsize=label_size,transform=ax.transAxes, zorder=500)
# #------------------------------------------------------------------------------
#### Ice
newcmap = cmocean.tools.crop_by_percent(cmocean.cm.ice, 20, which='min', N=None)
h3, ax2 = South_1ax_map(ax=ax2,
                      data=(X_full.loc[:,'ice']/X_full.loc[:,'ice'].max()).groupby(['latbins','lonbins']).mean().unstack('lonbins'),
                      plottype='mesh',
                      # levels=np.linspace(0, X_full.loc[:,'ice'].groupby(['latbins','lonbins']).mean().max(), 100),
                      vmin=0,
                      vmax=(X_full.loc[:,'ice']/X_full.loc[:,'ice'].max()).groupby(['latbins','lonbins']).mean().max()-0.5,
                      cmap=newcmap,
                      )

divider = make_axes_locatable(ax2)
ax_cb = divider.new_vertical(size="5%", pad=0.2, axes_class=plt.Axes, pack_start=True)
fig.add_axes(ax_cb)
cb = plt.colorbar(h3, cax=ax_cb, orientation="horizontal", extend='max')
cb.set_label(r'Fraction of Sea Ice Coverage', size=22)

ax2.text(-0.05,1,'$\mathbf{b}$',fontsize=label_size,transform=ax2.transAxes, zorder=500)

#------------------------------------------------------------------------------

#%% Temporal Correlations (Per Pixel)
if first_run == True:
    # pull out dates into columns - do this before the loops to speed up computations
    DMS_indexed = models_combined.unstack('datetime')
    vars_indexed = X_full.unstack('datetime')
    # create an empty list
    corrs_by_date = []
    # now iterate by predictor, computing correlations per coordinate over time (i.e. per row)
    for i,var_ in enumerate(X_full.columns):
        for j in tqdm(range(len(DMS_indexed))):
            corrs_by_date.append(spearmanr(DMS_indexed.iloc[j,:], vars_indexed.loc[:,var_].iloc[j,:])[0])
        if i == 0:
            corr_matrix = pd.Series(np.array(corrs_by_date), index=DMS_indexed.index, name=var_)
        else:
            iterated_var = pd.Series(np.array(corrs_by_date), index=DMS_indexed.index, name=var_)
            corr_matrix = pd.concat([corr_matrix, iterated_var], axis=1)
        corrs_by_date = []
    corr_matrix.to_csv(write_dir[:69]+'/'+'point_correlation_map_data.csv')
    del DMS_indexed, vars_indexed
else:
    corr_matrix = pd.read_csv(write_dir[:69]+'/'+'point_correlation_map_data.csv',index_col=[0,1], header=[0])

#%% Map correlations

sizing = 24
vmin = -1
vmax = 1
# Map predictors
for var in corr_matrix.columns:
    if var == 'PAR':
        fig = plt.figure(figsize=(18,18))
        font={'family':'DejaVu Sans',
              'weight':'normal',
              'size':'22'} 
        plt.rc('font', **font) # sets the specified font formatting globally
        gs = fig.add_gridspec(1, 1)
        # main plots
        ax = fig.add_subplot(gs[0,0], projection=ccrs.Orthographic(central_longitude=0, central_latitude=-90))
        norm = mpl.colors.TwoSlopeNorm(vmin=vmin,
                               vcenter=0,
                               vmax=vmax) # scales to accentuate depth colors, and diverge at 0
        h, ax = South_1ax_map(ax=ax,
                               data=corr_matrix.loc[:,var].unstack('lonbins'),
                              # data=corrs_to_map.loc[:,var].unstack('lonbins'),
                              plottype='mesh',
                              vmin=vmin,
                              vmax=vmax,
                              cmap='RdBu_r',
                              norm=norm,
                              )
        divider = make_axes_locatable(ax)
        ax_cb = divider.new_vertical(size="5%", pad=0.2, axes_class=plt.Axes, pack_start=True)
        fig.add_axes(ax_cb)
        cb = plt.colorbar(h, cax=ax_cb, orientation="horizontal")
        cb.ax.tick_params(labelsize=sizing)
        cb.set_label(var, fontsize=sizing)
        cb.set_ticks(np.arange(vmin, vmax, np.round(((vmax-vmin)/10),2)))
        # cb.set_ticklabels(np.arange(vmin, vmax+np.round(((vmax-vmin)/10),2), np.round(((vmax-vmin)/10),2)))

#%% PCA

#### Setup PCA
PCA_input = models_combined.unstack('datetime').T.reindex(index=reordered_months).dropna(axis=1) # transpose so eigenvectors are in space
PCA_scaler = StandardScaler()
PCA_scaler.fit(PCA_input.values)
data_norm = pd.DataFrame(PCA_scaler.transform(PCA_input.values), index=PCA_input.index, columns=PCA_input.columns)

#### Apply IPCA - runs PCA incrementally to reduce memory consumption
n_modes = np.min(np.shape(data_norm))
pca = IncrementalPCA(n_components = n_modes, batch_size=1000)
PCs = -1*pca.fit_transform(data_norm)
eigvecs = -1*pca.components_
fracVar = pca.explained_variance_ratio_

#### Plot Fraction of Variance per Mode
plt.figure(figsize=(18,12))
font={'family':'DejaVu Sans',
      'weight':'normal',
      'size':'22'} 
plt.rc('font', **font) # sets the specified font formatting globally
plt.subplot(1,1,1)
plt.plot(range(1,len(fracVar)+1),fracVar,'k--o',ms=10)
plt.xlabel('Mode Number')
plt.ylabel('Fraction Variance Explained')
plt.title('Variance Explained by All Modes')
plt.tight_layout()
plt.show()

#### Plot PCA spatial and temporal patterns
# choose number of modes to plot
n = 2

fig = plt.figure(figsize=(13*n,30))
font={'family':'DejaVu Sans',
      'weight':'normal',
      'size':'22'} 
plt.rc('font', **font) # sets the specified font formatting globally

for k in range(n):
    kPCs = PCs[:,k]
    if k==0:
        kPCs = -1*kPCs
    ax = fig.add_subplot(3,n,k+1)
    ax.plot(range(len(reordered_months)),kPCs,'k--')
    ax.scatter(range(len(reordered_months)),
                kPCs,
                s=300,
                c=PCs[:,k],
                edgecolors='k',
                cmap='RdBu_r',
                vmin=-600,
                vmax=700,
                zorder=10)
    ax.set_xticks(range(len(reordered_months)))
    ax.set_xticklabels(['Oct','Nov','Dec','Jan','Feb','Mar','Apr'])
    ax.set_ylim(-600,700)
    ax.set_title('PCs of Mode #' + str(k+1))
    ax.set_xlabel('Month')
    ax.text(0.05, 0.9, f'Variance = {fracVar[k]*100:.2f}%', transform=ax.transAxes, fontsize=22,
            va='center', ha='left', ma='left', zorder=500)
    
    norm = mpl.colors.TwoSlopeNorm(vmin=eigvecs[:n,:].min(),
                                    vcenter=0,
                                    vmax=eigvecs[:n,:].max()) # scales to accentuate depth colors, and diverge at 0
    ax2 = fig.add_subplot(3,n,(n)+k+1,projection=ccrs.Orthographic(central_longitude=0, central_latitude=-90))
    keigvecs = pd.DataFrame(eigvecs[k,:].T, index=PCA_input.columns).squeeze()
    if k==0:
        keigvecs = -1*keigvecs
    h, ax2 = South_1ax_map(ax=ax2,
                          data=keigvecs.unstack('lonbins'),
                          plottype='mesh',
                          vmin=eigvecs[:n,:].min(),
                          vmax=eigvecs[:n,:].max(),
                          norm=norm,
                          cmap='RdBu_r',
                          )
    ax2.set_title('Eigenvectors of Mode #' + str(k+1))
    if k == 1:
        divider = make_axes_locatable(ax2)
        ax_cb = divider.new_horizontal(size="5%", pad=0.6, axes_class=plt.Axes)
        fig.add_axes(ax_cb)
        cb1 = plt.colorbar(h, cax=ax_cb)
        cb1.ax.tick_params(labelsize=20)
        cb1.set_ticks(np.linspace(np.round(eigvecs[:n,:].min(),3),np.round(eigvecs[:n,:].max(),3),5))
        cb1.set_label('$\it{v}$', fontsize=20)
    
    ax2.scatter(x=fronts['PF'].index.get_level_values('lonbins').values,
                y=fronts['PF'].index.get_level_values('latbins').values,
                s=fronts['PF'].values,
                c='k',
                transform=ccrs.PlateCarree())
    
    ax3 = fig.add_subplot(3,n,(2*n)+k+1,projection=ccrs.Orthographic(central_longitude=0, central_latitude=-90))
    names = ['MLD','SST','SAL']
    h2, ax3 = South_1ax_map(ax=ax3,
                            data=corr_matrix.loc[:,names[k]].unstack('lonbins'),
                            plottype='mesh',
                            vmin=-1,
                            vmax=1,
                            cmap='RdBu_r',
                            )
    ax3.set_title(names[k])
    if k == 1:
        divider = make_axes_locatable(ax3)
        ax_cb = divider.new_horizontal(size="5%", pad=0.6, axes_class=plt.Axes)
        fig.add_axes(ax_cb)
        cb1 = plt.colorbar(h2, cax=ax_cb)
        cb1.ax.tick_params(labelsize=20)
        cb1.set_label(r'$\rho$', fontsize=20)
    ax3.scatter(x=fronts['PF'].index.get_level_values('lonbins').values,
                y=fronts['PF'].index.get_level_values('latbins').values,
                s=fronts['PF'].values,
                c='k',
                transform=ccrs.PlateCarree())

fig.subplots_adjust(hspace=0.2)

#%% Plot Mesoscale variability at Kerguelen Region
label_size = 32

fig = plt.figure(figsize=(40,22))
font={'family':'DejaVu Sans',
      'weight':'normal',
      'size':'24'} 
plt.rc('font', **font) # sets the specified font formatting globally
gs = fig.add_gridspec(4, 5)
# main plots
# Increase resolution of projection - needed to draw polygons accurately
map_proj = ccrs.Orthographic(central_latitude=-90.0, central_longitude=0)
map_proj._threshold /= 100
ax = fig.add_subplot(gs[0:3, 0:2], projection=map_proj)
ax3 = fig.add_subplot(gs[0:2, 2:5], projection=ccrs.PlateCarree())
ax4 = fig.add_subplot(gs[2:5, 2:5], projection=ccrs.PlateCarree())
ax5 = fig.add_subplot(gs[3, 0], projection=ccrs.PlateCarree())
ax6 = fig.add_subplot(gs[3, 1], projection=ccrs.PlateCarree())

#------------------------------------------------------------------------------

extent = [60, 90, -54, -40]
newextent = [67, 77, -54, -46]
month = 1

#------------------------------------------------------------------------------
#### Plot SSHA correlations
h, ax = South_1ax_map(ax=ax,
                      data=corr_matrix.loc[:,'SSHA'].unstack('lonbins'),
                      # data=corr_matrix.loc[:,'SSHA'].where((corr_matrix.loc[:,'SSHA']>0.5) | (corr_matrix.loc[:,'SSHA']<-0.5)).unstack('lonbins'),
                      plottype='mesh',
                      vmin=corr_matrix.loc[:,'SSHA'].unstack('lonbins').loc[extent[2]:extent[3],extent[0]:extent[1]].min().min(),
                      vmax=corr_matrix.loc[:,'SSHA'].unstack('lonbins').loc[extent[2]:extent[3],extent[0]:extent[1]].max().max(),
                      cmap='RdBu_r')

ax.gridlines(draw_labels=True,
            lw=3,
            color="k",
            y_inline=True,
            xlocs=range(-180,180,30),
            ylocs=range(-80,91,10),
            zorder=50,
            )

divider = make_axes_locatable(ax)
ax_cb = divider.new_vertical(size="5%", pad=0.12, axes_class=plt.Axes, pack_start=True)
fig.add_axes(ax_cb)
cb1 = plt.colorbar(h, cax=ax_cb, orientation='horizontal')
cb1.ax.tick_params(labelsize=20)
cb1.set_label(r'$\rho$(DMS, SSHA)', fontsize=20)

ax.text(-0.05,1,'$\mathbf{a}$',fontsize=label_size,transform=ax.transAxes, zorder=500)

def custom_mark_zoom(axA, axB, direction='right', extent=None, fc=None, ec='k', alpha=1, transform=None):
    # starting point:
    # https://stackoverflow.com/questions/51268493/drawing-filled-shapes-between-different-axes-in-matplotlib
    import matplotlib.patches as patches
    import numpy as np
    import matplotlib as mpl
    
    xx = [extent[0], extent[1]]
    yy = [extent[2], extent[3]]
    xy = (xx[0], yy[0])
    width = xx[1] - xx[0]
    height = yy[1] - yy[0]
    
    xyB1 = (0,1)
    xyB2 = (0,0)
    xyA1 = transform.transform_point(60,-40,ccrs.PlateCarree())
    xyA2 = transform.transform_point(90,-40,ccrs.PlateCarree())
    
    coordsA='data'
    coordsB='axes fraction'
    
    # First mark the patch in the main axes
    pp = axA.add_patch(patches.Rectangle(xy, width, height, fc=fc, ec=ec, zorder=5, alpha=alpha, transform=ccrs.PlateCarree()))
    # Add a second identical patch w/o alpha & face color (i.e. make the edge color dark)
    pp = axA.add_patch(patches.Rectangle(xy, width, height, fc='None', ec=ec, lw=2, zorder=5, transform=ccrs.PlateCarree()))
    
    # now draw an anchor line to the zoomed in axis
    p1 = axA.add_patch(patches.ConnectionPatch(
        xyA=xyA1, xyB=xyB1,
        coordsA=coordsA, coordsB=coordsB,
        axesA=axA, axesB=axB))
    
    # draw a 2nd anchor line to the zoomed in axes
    p2 = axA.add_patch(patches.ConnectionPatch(
        xyA=xyA2, xyB=xyB2,
        coordsA=coordsA, coordsB=coordsB,
        axesA=axA, axesB=axB))
        
    return pp, p1, p2

# add the connection lines and shading
pp, p1, p2 = custom_mark_zoom(ax, ax3, direction='right', extent=[60, 90, -54, -40], fc='gray', alpha=0.5, transform=map_proj)

#------------------------------------------------------------------------------
#### Plot DMS subregion
h2, ax3, gl = South_1ax_flat_map(ax=ax3,
                        data=models_combined.loc[month].unstack('lonbins'),
                        plottype='mesh',
                        # levels=100,
                        vmin=0,
                        vmax=10,
                        # cmap='viridis',
                        cmap=cmocean.cm.haline,
                        extent=extent)
gl.bottom_labels = False

var = 'SSHA'
h0 = ax3.contour(X_full.loc[:,var].loc[month].unstack('lonbins').loc[extent[2]-3:extent[3],extent[0]:extent[1]].columns.values,
                X_full.loc[:,var].loc[month].unstack('lonbins').loc[extent[2]-3:extent[3],extent[0]:extent[1]].index.values,
                X_full.loc[:,var].loc[month].unstack('lonbins').loc[extent[2]-3:extent[3],extent[0]:extent[1]].values,
                levels=[-0.25,-0.2,-0.15,-0.1,-0.05,0.05,0.1,0.15,0.2,0.25],
                # levels=10,
                colors='k',
                linewidths=0.5,
                transform=ccrs.PlateCarree())
ax3.clabel(h0, fontsize=10)

# Plot PF and SAF fronts
ax3.plot(front_data.LonPF.values,
        front_data.LatPF.values,
        'r-',
        linewidth=3,
        transform=ccrs.PlateCarree())
ax3.plot(front_data.LonSAF.values,
        front_data.LatSAF.values,
        'w-',
        linewidth=3,
        transform=ccrs.PlateCarree())

# Contour topography
h01 = ax3.contour(etopo.unstack('lonbins').loc[extent[2]-3:extent[3],extent[0]:extent[1]].columns.values,
                etopo.unstack('lonbins').loc[extent[2]-3:extent[3],extent[0]:extent[1]].index.values,
                etopo.unstack('lonbins').loc[extent[2]-3:extent[3],extent[0]:extent[1]].values,
                levels=[-1500],
                colors='w',
                linestyles='dashed',
                linewidths=2,
                transform=ccrs.PlateCarree())
for c in h01.collections:
    c.set_dashes([(0, (2.0, 5.0))])

ax3.set_title(f'Kerguelen Plateau ({var_months_[month]})')
divider = make_axes_locatable(ax3)
ax_cb = divider.new_horizontal(size="5%", pad=0.2, axes_class=plt.Axes)
fig.add_axes(ax_cb)
cb1 = plt.colorbar(h2, cax=ax_cb, extend='max')
cb1.ax.tick_params(labelsize=20)
cb1.set_label(r'DMS$_{\rmmodel}$ (nM)')

# Add legend
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

legend_elements = [Line2D([0], [0], color='r', label='Polar Front (PF)'),
                   Line2D([0], [0], color='w', label='Southern Antarctic Front (SAF)'),
                   ]
ax3.legend(handles=legend_elements, loc='lower left',framealpha=0.7)

ax3.text(-0.05,1,'$\mathbf{b}$',fontsize=label_size,transform=ax3.transAxes, zorder=500)

#------------------------------------------------------------------------------
#### Plot 2nd variable for subregion
mapvar = 'chl'
h3, ax4, _ = South_1ax_flat_map(ax=ax4,
                        data=X_full.loc[:,mapvar].loc[month].unstack('lonbins'),
                        plottype='mesh',
                        # levels=100,
                        vmin=X_full.loc[:,mapvar].loc[1,extent[2]-3:extent[3],extent[0]:extent[1]].min(),
                        vmax=X_full.loc[:,mapvar].loc[1,extent[2]-3:extent[3],extent[0]:extent[1]].max(),
                        cmap=cmocean.cm.thermal,
                        extent=extent)

var = 'SSHA'
h0 = ax4.contour(X_full.loc[:,var].loc[month].unstack('lonbins').loc[extent[2]-3:extent[3],extent[0]:extent[1]].columns.values,
                  X_full.loc[:,var].loc[month].unstack('lonbins').loc[extent[2]-3:extent[3],extent[0]:extent[1]].index.values,
                  X_full.loc[:,var].loc[month].unstack('lonbins').loc[extent[2]-3:extent[3],extent[0]:extent[1]].values,
                  levels=[-0.25,-0.2,-0.15,-0.1,-0.05,0.05,0.1,0.15,0.2,0.25],
                  # levels=15,
                  colors='w',
                  linewidths=0.5,
                  transform=ccrs.PlateCarree())
ax4.clabel(h0, fontsize=10)
 
# mark the inset box
ax4.add_patch(mpl.patches.Rectangle(xy=(newextent[0], newextent[2]), width=newextent[1]-newextent[0], height=newextent[3]-newextent[2],
                                    ec='r',
                                    linestyle='-',
                                    lw=4,
                                    fill=False,
                                    alpha=1,
                                    zorder=1000,
                                    transform=ccrs.PlateCarree()))

divider = make_axes_locatable(ax4)
ax_cb = divider.new_horizontal(size="5%", pad=0.2, axes_class=plt.Axes)
fig.add_axes(ax_cb)
cb1 = plt.colorbar(h3, cax=ax_cb, extend='both')
cb1.ax.tick_params(labelsize=20)
cb1.set_label(r'chl-a (mg m$^{-3}$)')

# Plot PF and SAF fronts
h0 = ax4.plot(front_data.LonPF.values,
              front_data.LatPF.values,
              'r-',
              linewidth=3,
              transform=ccrs.PlateCarree())
h0 = ax4.plot(front_data.LonSAF.values,
              front_data.LatSAF.values,
              'w-',
              linewidth=3,
              transform=ccrs.PlateCarree())

# Contour topography
h01 = ax4.contour(etopo.unstack('lonbins').loc[extent[2]-3:extent[3],extent[0]:extent[1]].columns.values,
                etopo.unstack('lonbins').loc[extent[2]-3:extent[3],extent[0]:extent[1]].index.values,
                etopo.unstack('lonbins').loc[extent[2]-3:extent[3],extent[0]:extent[1]].values,
                levels=[-1500],
                colors='w',
                linestyles='dashed',
                linewidths=2,
                transform=ccrs.PlateCarree())
for c in h01.collections:
    c.set_dashes([(0, (2.0, 5.0))])

ax4.text(-0.05,1,'$\mathbf{c}$',fontsize=label_size, transform=ax4.transAxes, zorder=500)
#------------------------------------------------------------------------------
#### Plot CDOM over plateau
h2, ax5, _ = South_1ax_flat_map(ax=ax5,
                               data=X_full.loc[month,'CDOM'].unstack('lonbins'),
                               plottype='mesh',
                               cmap=cmocean.cm.thermal,
                               vmin=np.nanmin(X_full.loc[month,'CDOM'].loc[newextent[2]:newextent[3],newextent[0]:newextent[1]]),
                               vmax=np.nanmax(X_full.loc[month,'CDOM'].loc[newextent[2]:newextent[3],newextent[0]:newextent[1]]),
                               extent=newextent
                               )
divider = make_axes_locatable(ax5)
ax_cb = divider.new_horizontal(size="5%", pad=0.2, axes_class=plt.Axes)
fig.add_axes(ax_cb)
cb1 = plt.colorbar(h2, cax=ax_cb, extend='both')
cb1.ax.tick_params(labelsize=20)
cb1.set_label(r'a$_{443}$ (m$^{-1}$)')

# Plot PF and SAF fronts
h0 = ax5.plot(front_data.LonPF.values,
              front_data.LatPF.values,
              'r-',
              linewidth=3,
              transform=ccrs.PlateCarree())
h0 = ax5.plot(front_data.LonSAF.values,
              front_data.LatSAF.values,
              'w-',
              linewidth=3,
              transform=ccrs.PlateCarree())

# Contour topography
h01 = ax5.contour(etopo.unstack('lonbins').loc[extent[2]-3:extent[3],extent[0]:extent[1]].columns.values,
                etopo.unstack('lonbins').loc[extent[2]-3:extent[3],extent[0]:extent[1]].index.values,
                etopo.unstack('lonbins').loc[extent[2]-3:extent[3],extent[0]:extent[1]].values,
                levels=[-1500],
                colors='w',
                linestyles='dashed',
                linewidths=2,
                transform=ccrs.PlateCarree())
for c in h01.collections:
    c.set_dashes([(0, (2.0, 5.0))])

ax5.text(-0.1,1,'$\mathbf{d}$',fontsize=label_size,transform=ax5.transAxes, zorder=500)

#------------------------------------------------------------------------------
#### Plot SSN over plateau
h2, ax6, gl = South_1ax_flat_map(ax=ax6,
                               data=X_full.loc[month,'SSN'].unstack('lonbins'),
                               plottype='mesh',
                               cmap=cmocean.cm.thermal,
                               vmin=np.nanmin(X_full.loc[month,'SSN'].loc[newextent[2]:newextent[3],newextent[0]:newextent[1]]),
                               vmax=np.nanmax(X_full.loc[month,'SSN'].loc[newextent[2]:newextent[3],newextent[0]:newextent[1]]),
                               extent=newextent
                               )
gl.left_labels = False
divider = make_axes_locatable(ax6)
ax_cb = divider.new_horizontal(size="5%", pad=0.2, axes_class=plt.Axes)
fig.add_axes(ax_cb)
cb1 = plt.colorbar(h2, cax=ax_cb, extend='both')
cb1.ax.tick_params(labelsize=20)
cb1.set_label(r'SSN ($\mathrm{\mu}$mol $\mathrm{kg^{-1}}$)')

# Plot PF and SAF fronts
h0 = ax6.plot(front_data.LonPF.values,
              front_data.LatPF.values,
              'r-',
              linewidth=3,
              transform=ccrs.PlateCarree())
h0 = ax6.plot(front_data.LonSAF.values,
              front_data.LatSAF.values,
              'w-',
              linewidth=3,
              transform=ccrs.PlateCarree())

# Contour topography
h01 = ax6.contour(etopo.unstack('lonbins').loc[extent[2]-3:extent[3],extent[0]:extent[1]].columns.values,
                etopo.unstack('lonbins').loc[extent[2]-3:extent[3],extent[0]:extent[1]].index.values,
                etopo.unstack('lonbins').loc[extent[2]-3:extent[3],extent[0]:extent[1]].values,
                levels=[-1500],
                colors='w',
                linestyles='dashed',
                linewidths=2,
                transform=ccrs.PlateCarree())
for c in h01.collections:
    c.set_dashes([(0, (2.0, 5.0))])

ax5.text(-0.1,1,'$\mathbf{e}$',fontsize=label_size,transform=ax6.transAxes, zorder=500)

fig.subplots_adjust(hspace=0.3)



#%% Spatial Correlations
# Use spearman rank to assess non-linear relationships
X_full_plus = X_full.copy()
X_full_plus['Si_star'] = Si_star
X_plus = X.copy()
X_plus['Si_star'] = (X.loc[:,'Si']-X.loc[:,'SSN']).squeeze()
#-----------------------------------------------------------------------------
#### RFR correlations
rhos = []
pvals = []
names = X_full_plus.columns.values
for i in range(len(X_full_plus.columns.values)):
    rho, pval = spearmanr(X_full_plus.iloc[:,i].values,np.sinh(RFR_y_pred.values))
    rhos.append(rho)
    pvals.append(pval)
RFR_correlations = pd.concat([pd.Series(names),pd.Series(rhos),pd.Series(np.array(rhos)**2),pd.Series(pvals)], axis=1)
RFR_correlations.columns = ['Variable','rho','R2','p-value']
RFR_correlations.set_index('Variable', inplace=True)
# RFR_correlations = RFR_correlations.reindex(RFR_correlations.rho.abs().sort_values(ascending=False).index)
#-----------------------------------------------------------------------------
#### ANN correlations
rhos = []
pvals = []
names = X_full_plus.columns.values
for i in range(len(X_full_plus.columns.values)):
    rho, pval = spearmanr(X_full_plus.iloc[:,i].values,np.sinh(ANN_y_pred.values))
    rhos.append(rho)
    pvals.append(pval)
ANN_correlations = pd.concat([pd.Series(names),pd.Series(rhos),pd.Series(np.array(rhos)**2),pd.Series(pvals)], axis=1)
ANN_correlations.columns = ['Variable','rho','R2','p-value']
ANN_correlations.set_index('Variable', inplace=True)
# ANN_correlations = ANN_correlations.reindex(ANN_correlations.rho.abs().sort_values(ascending=False).index)
#------------------------------------------------------------------------------
#### Raw Data
rhos = []
pvals = []
names = X_plus.columns.values
for i in range(len(X_plus.columns.values)):
    rho, pval = spearmanr(X_plus.iloc[:,i].values,np.sinh(y.values))
    rhos.append(rho)
    pvals.append(pval)
correlations = pd.concat([pd.Series(names),pd.Series(rhos),pd.Series(np.array(rhos)**2),pd.Series(pvals)], axis=1)
correlations.columns = ['Variable','rho','R2','p-value']
correlations.set_index('Variable', inplace=True)
# correlations = correlations.reindex(correlations.rho.abs().sort_values(ascending=False).index)
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
print(RFR_correlations)
print(ANN_correlations)
print(correlations)
#------------------------------------------------------------------------------
corr_mat = pd.concat([correlations['rho'],RFR_correlations['rho'],ANN_correlations['rho']],axis=1)
corr_mat.columns=['Obs.','RFR','ANN']
corr_mat=corr_mat.T
#### Plot heatmap
fig = plt.figure(figsize=(24,24))
font = {'family' : 'DejaVu Sans',
        'weight' : 'normal',
        'size'   : 22}
mpl.rc('font', **font)
ax = fig.add_subplot(111)

# Set up diverging colormaps...
# colors1 = plt.cm.gist_heat(np.linspace(0, 1, 128))
# colors2 = plt.cm.gist_heat_r(np.linspace(0, 1, 128))
colors1 = plt.cm.Blues_r(np.linspace(0, 1, 128))
colors2 = plt.cm.Reds(np.linspace(0, 1, 128))
# ...combine them and build a new colormap
colors = np.vstack((colors1, colors2))
cmap = mpl.colors.LinearSegmentedColormap.from_list('my_colormap', colors)

# normalize cmap to diverge at 0
norm = mpl.colors.TwoSlopeNorm(vmin=-0.4, vcenter=0, vmax=0.4)

# plot data
im = ax.imshow(corr_mat, cmap=cmap, norm=norm)

# We want to show all ticks...
ax.set_xticks(np.arange(len(corr_mat.columns.values)))
ax.set_yticks(np.arange(len(corr_mat.index.values)))
# ... and label them with the respective list entries
ax.set_xticklabels(corr_mat.columns.values)
ax.set_yticklabels(corr_mat.index.values)

# Rotate the tick labels and set their alignment
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations
for i in range(len(corr_mat.index.values)):
    for j in range(len(corr_mat.columns.values)):
        if corr_mat.iloc[i, j]<0.1 and corr_mat.iloc[i, j]>-0.1:
            text = ax.text(j, i, round(corr_mat.iloc[i, j], ndigits=2),
                           ha="center", va="center", color="k")
        else:
            text = ax.text(j, i, round(corr_mat.iloc[i, j], ndigits=2),
                       ha="center", va="center", color="w")
ax.set_title(r'Spearman Rank Correlations ($\rho$)')
# plt.colorbar(im, ax=ax, shrink=0.2)
divider = make_axes_locatable(ax)
ax_cb2 = divider.new_horizontal(size="3%", pad=0.2, axes_class=plt.Axes)
fig.add_axes(ax_cb2)
cb2 = plt.colorbar(im, cax=ax_cb2)
cb2.ax.tick_params(labelsize=20)
cb2.set_label(r'$\rho$')

# fig.savefig(save_to_path+str('heatmap.png'), dpi=500, transparent=True, bbox_inches='tight')

