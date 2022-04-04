Scripts included here generate a boreal summertime DMS climatology for the Southern Ocean, using ensembled random forest regression and artificial neural network algorithms.

The main scripts include: 
* SO_DMS_build_models.py: main script processing/interpolating data and building the RFR and ANN ensembles.
* SO_DMS_analysis.py: main script for analysis, with code computing flux calculations and generating figures

Ancillary scripts with subfunctions including sea-air flux calculations, the neural network design, and mapping templates are also included. 

All DMS data can be found in the NOAA PMEL repository (https://saga.pmel.noaa.gov/dms/). For processing steps applied to predictor data, see "Satellite_data_processing_SO_35gb.ipynb".

PLEASE NOTE: The "Taylordiagram.py" script includes functions to generate a Taylor Diagram (Taylor, 2001). It is from the public domain and is NOT my creation - all credit goes to Yannick Copin (https://gist.github.com/ycopin/3342888). However, this version is included for compatability and is modified to do the following:

*includes functionality that enables the user to normalize the standard deviations (and RMSE contours) to the reference data (i.e. the reference point is at a standard deviation of 1.0). This also replaces the x-axis label with "Standard deviation (norm.)" when enabled.
*includes functionality that enables a legend and text boxes to be added to the figure.
*changes the correlation tick values/locations.
*changes the reference point to a gold star and increases it's size.

Relevant publication: McNabb & Tortell. Physical Controls on Southern Ocean Dimethyl Sulfide (DMS) Distributions Revealed by Machine Learning Algorithms (2022). Limnology and Oceanography, in review.
