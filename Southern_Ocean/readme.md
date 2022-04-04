Scripts included here generate a boreal summertime DMS climatology for the Southern Ocean, using ensembled random forest regression and artificial neural network algorithms.

The main scripts include: 
* SO_DMS_build_models.py: main script processing/interpolating data and building the RFR and ANN ensembles.
* SO_DMS_analysis.py: main script for analysis, with code computing flux calculations and generating figures

See "Ancillary_Scripts" for subfunctions including processing steps applied to predictor data, sea-air flux calculations, the neural network design, and mapping templates. 

All DMS data can be found in the NOAA PMEL repository (https://saga.pmel.noaa.gov/dms/).

Relevant publication: McNabb & Tortell. Physical Controls on Southern Ocean Dimethyl Sulfide (DMS) Distributions Revealed by Machine Learning Algorithms (2022). Limnology and Oceanography, in review.
![DMS_timeseries](https://user-images.githubusercontent.com/68400556/161633959-1ebbbef7-d62e-46d0-a7e0-d35cf432577e.gif)
