# -*- coding: utf-8 -*-
"""
Created on Sun Sep 19 17:39:51 2021

@author: bcamc
"""
#%% Import packages

import cartopy.io.shapereader as shpreader
from cartopy.feature import ShapelyFeature
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs

#%% Define some plotting functions
#-----------------------------------------------------------------------------

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    
    see: https://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points
    """
    from math import radians, cos, sin, asin, sqrt
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r

#-----------------------------------------------------------------------------
# Import in shape files for ice
glaciers = shpreader.Reader('C:/Users/bcamc/OneDrive/Desktop/Python/Projects/sulfur/southern_ocean/ice_shapefiles/ne_10m_glaciated_areas/ne_10m_glaciated_areas.shp')
ice_shelves = shpreader.Reader('C:/Users/bcamc/OneDrive/Desktop/Python/Projects/sulfur/southern_ocean/ice_shapefiles/ne_10m_antarctic_ice_shelves_polys/ne_10m_antarctic_ice_shelves_polys.shp')
ice_shelves_edges = shpreader.Reader('C:/Users/bcamc/OneDrive/Desktop/Python/Projects/sulfur/southern_ocean/ice_shapefiles/ne_10m_antarctic_ice_shelves_lines/ne_10m_antarctic_ice_shelves_lines.shp')
#-----------------------------------------------------------------------------
# Mapping template:
def South_1ax_map(ax=None, data=None, plottype='mesh', cmap=None, vmin=0, vmax=10, levels=100, norm=None, extend=None, s=None, colors=None):
    """
    Plots a polar map of the Southern Ocean (<40oS) on a single axis.

    Parameters
    ----------
    data : DataFrame, optional
        A multiindex column dataframe with a specified index of "datetime", "latbins", and "lonbins". The default is None.
    plottype : bool, optional
        Choose whether to plot data as a colored grid ("mesh" = pcolormesh), contours ("contour"), filled contours ("contourf") or scatter plots ("scatter"). The default is 'mesh'.
    cmap : str, optional
        A matplotlib colormap. The default is 'viridis'.
    vmin : float, optional
        The colorbar range minimum. The default is 0.
    vmax : float, optional
        The colorbar range maximum. The default is 10.
    levels : int, optional
        Sets number of contour lines. The default is 100.

    Returns
    -------
    ax : ax
        The matplotlib axis generated.

    """
    if norm != None:
        norm=norm
    if extend != None:
        extend=extend
    if s != None:
        s=s
    if colors != None:
        colors=colors
    if cmap != None:
        cmap=cmap
    if ax is None:
        # Generate figure and axes handles
        fig = plt.figure(figsize=(18,18))
        font={'family':'DejaVu Sans',
              'weight':'normal',
              'size':'22'} 
        plt.rc('font', **font) # sets the specified font formatting globally
        gs = fig.add_gridspec(1, 1)
        # Increase resolution of projection - needed to draw polygons accurately
        map_proj = ccrs.Orthographic(central_latitude=-90.0, central_longitude=0)
        map_proj._threshold /= 100
        # main plots
        ax = fig.add_subplot(gs[0,0], projection=map_proj)

    # add features
    ax.add_feature(ShapelyFeature(ice_shelves.geometries(),
                                    ccrs.PlateCarree(), facecolor='aliceblue', edgecolor='black', zorder=1))
    ax.add_feature(ShapelyFeature(glaciers.geometries(),
                                    ccrs.PlateCarree(), facecolor='lightblue', edgecolor='black', zorder=1))
    ax.add_feature(ShapelyFeature(ice_shelves_edges.geometries(),
                                    ccrs.PlateCarree(), facecolor='azure', edgecolor='black', zorder=1))
    ax.add_feature(cartopy.feature.LAND, edgecolor='None', facecolor='darkgray', zorder=2)
    ax.add_feature(cartopy.feature.COASTLINE, edgecolor='k', zorder=2)
    
    # plot data
    if data is None:
        h = None
        pass
    else:
        if plottype=='contourf':
            h = ax.contourf(data.columns.values,
                            data.index.get_level_values('latbins').values,
                            data.values,
                            cmap=cmap,
                            levels=levels,
                            vmin = vmin, vmax = vmax,
                            norm=norm,
                            extend=extend,
                            transform=ccrs.PlateCarree())
        elif plottype=='contour':
            h = ax.contour(data.columns.values,
                           data.index.get_level_values('latbins').values,
                           data.values,
                           vmin=None,
                           vmax=None,
                           levels=levels,
                           colors=colors,
                           cmap=None,
                           transform=ccrs.PlateCarree())
        elif plottype=='mesh':
            h = ax.pcolormesh(data.columns.values,
                              data.index.get_level_values('latbins').values,
                              data.values,
                              shading='nearest',
                              cmap=cmap,
                              vmin = vmin, vmax = vmax,
                              norm=norm,
                              transform=ccrs.PlateCarree())
        elif plottype=='scatter':
            h = ax.scatter(x=data.stack(dropna=False).index.get_level_values('lonbins').values,
                           y=data.stack(dropna=False).index.get_level_values('latbins').values,
                           c=data.values,
                           s=s,
                           cmap=cmap,
                           vmin = vmin, vmax = vmax,
                           norm=norm,
                           transform=ccrs.PlateCarree())
    
    ax.set_extent([-180,180,-90,-30], crs=ccrs.PlateCarree())
    # Create a circular map frame bounded by radius of map extent:
    # see https://stackoverflow.com/questions/67877552/how-to-zoom-into-a-specific-latitude-in-cartopy-crs-orthographic
    import matplotlib.path as mpath
    r_limit = haversine(0,-90,0,-46)*1000 # convert km to m for CartoPy
    circle_path = mpath.Path.unit_circle()
    circle_path = mpath.Path(circle_path.vertices.copy() * r_limit,
                                circle_path.codes.copy())
    # set circle boundary & extent
    ax.set_boundary(circle_path)
    
    # add gridlines
    gl = ax.gridlines(draw_labels=True,
                      lw=3,
                      color="silver",
                      y_inline=True,
                      xlocs=range(-180,180,30),
                      ylocs=range(-80,91,10),
                      zorder=50,
                      )
    return h, ax

def South_1ax_flat_map(ax=None, data=None, plottype='mesh', cmap='viridis', vmin=0, vmax=10, levels=100, cm=180, extent=None, norm=None, extend=None):
    """
    Plots a PlateCarree map of the Southern Ocean (<40oS) on a single axis.

    Parameters
    ----------
    data : DataFrame, optional
        A multiindex column dataframe with a specified index of "datetime", "latbins", and "lonbins". The default is None.
    plottype : bool, optional
        Choose whether to plot data as a colored grid ("mesh" = pcolormesh) or filled contours ("contour" = contourf). The default is 'mesh'.
    cmap : str, optional
        A matplotlib colormap. The default is 'viridis'.
    vmin : float, optional
        The colorbar range minimum. The default is 0.
    vmax : float, optional
        The colorbar range maximum. The default is 10.
    levels : TYPE, optional
        Sets number of contour lines. The default is 100.
    cm: float
        Sets the central longitude of the projection. Change between 0 and 180
        if plotting a subset of coordinates intersected by the meridian.

    Returns
    -------
    ax : ax
        The matplotlib axis generated.

    """
    if norm != None:
        norm=norm
    if extend != None:
        extend=extend
    if ax is None:
        # Generate figure and axes handles
        fig = plt.figure(figsize=(18,18))
        font={'family':'DejaVu Sans',
              'weight':'normal',
              'size':'22'} 
        plt.rc('font', **font) # sets the specified font formatting globally
        gs = fig.add_gridspec(1, 1)
        # main plots
        ax = fig.add_subplot(gs[0,0], projection=ccrs.PlateCarree(central_longitude=cm))
    #-----------------------------------------------------------------------------
    # Plot measured DMS
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    if extent is None:
        ax.set_extent([-179.999, 180, -90, -40])
    else:
        ax.set_extent(extent)

    # add features
    ax.add_feature(ShapelyFeature(ice_shelves.geometries(),
                                    ccrs.PlateCarree(), facecolor='aliceblue', edgecolor='black', zorder=1))
    ax.add_feature(ShapelyFeature(glaciers.geometries(),
                                    ccrs.PlateCarree(), facecolor='lightblue', edgecolor='black', zorder=1))
    ax.add_feature(ShapelyFeature(ice_shelves_edges.geometries(),
                                    ccrs.PlateCarree(), facecolor='azure', edgecolor='black', zorder=1))
    ax.add_feature(cartopy.feature.LAND, edgecolor='None', facecolor='darkgray', zorder=2)
    ax.add_feature(cartopy.feature.COASTLINE, edgecolor='k', zorder=2)
    
    if data is None:
        h = None
        pass
    else:
        if plottype=='contourf':
            h = ax.contourf(data.columns.values,
                            data.index.get_level_values('latbins').values,
                            data.values,
                            cmap=cmap,
                            levels=levels,
                            vmin = vmin, vmax = vmax,
                            norm=norm,
                            extend=extend,
                            transform=ccrs.PlateCarree())
        elif plottype=='mesh':
            h = ax.pcolormesh(data.columns.values,
                              data.index.get_level_values('latbins').values,
                              data.values,
                              shading='nearest',
                              cmap=cmap,
                              vmin = vmin, vmax = vmax,
                              norm=norm,
                              transform=ccrs.PlateCarree())
    return h, ax, gl

