# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 17:52:20 2022

@author: Brandon McNabb (bmcnabb@eoas.ubc.ca)
"""

def calculate_fluxes(data, ice_cover, wind_speed, T, parameterization='GM12'):
    import scipy
    import numpy as np
    if parameterization=='GM12':
        # Goddijn-Murphy et al. (2012) parameterization
        # Gas transfer velocity (cm^3 hr^-1) for DMS:
        k_dms = (2.1*wind_speed)-2.8
        # Proportion of ice cover
        ice_frac = ice_cover/np.nanmax(ice_cover)
        #-----------------------------------------------------------------------------
        # Fluxes (umol m^-2 d^-1):
        flux = k_dms*data*((1-ice_frac)**0.4)*0.24 # 0.24 converts hr^-1 to d^-1 (& cm to m)
    elif parameterization=='SD02':
        # See Simo & Dachs (2002)
        Xi = 2 # Assuming a Rayleigh Distribution, see Livingstone & Imboden (1993)
        s = (1+(1/Xi))
        eta_sq = (wind_speed/scipy.special.gamma(s))**2
        # Schmidt number (cm^2 sec^-1):
        Sc = 2674-(147.12*T)+(3.72*T**2)-(0.038*T**3)
        # Gas transfer velocity (cm^3 hr^-1) for DMS:
        k_dms = (5.88*eta_sq*scipy.special.gamma(1+(2/Xi))\
                  +1.49*np.sqrt(eta_sq)*scipy.special.gamma(s))*(Sc**-0.5) # See Simo & Dachs (2007)
        # Proportion of ice cover
        ice_frac = ice_cover/np.nanmax(ice_cover)
        #-----------------------------------------------------------------------------
        # Fluxes (umol m^-2 d^-1):
        flux = k_dms*data*((1-ice_frac)**0.4)*0.24 # 0.24 converts hr^-1 to d^-1 (& cm to m)
    return k_dms, flux
        
        
        
