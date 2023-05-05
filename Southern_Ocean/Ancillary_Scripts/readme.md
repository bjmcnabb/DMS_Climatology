![PyTorch_ANN_Fit](https://user-images.githubusercontent.com/68400556/161632855-8fa55e2e-be69-47d4-94a2-f52e9ad1a0eb.gif)

Scripts here provide the following:
* NN_model_frameworks.py: includes a PyTorch class that builds an artificial neural network for regression. Subfunctions include functionality to convert numpy arrays to torch datasets for compatability and to produce a GIF of the training process (i.e. plots both model fit and loss for each training epoch - see above). Updated to include an experimental custom ensembling function modelled after the ensembletorch library, but using multithreading parallization on predictions (essentially scikit-learn's implementation on RFR) to speed up prediction times. Performance is currently below the packaged torchensemble equivalent.

* SO_mapping_templates.py: boilerplate to produce orthographic and plate carree projections of the Southern Ocean (below 40oS), with formatting to plot the location of relevant glaciers and ice shelves. The subfunctions include functionality for plotting contours (filled or unfilled), pcolormesh, or scatterplots.
* Fluxes.py: function for computing sea-air fluxes of DMS, using either the GM12 or SD02 parameterizations (see the associated manuscripts; Goddijin-Murphy et al. 2012, Simo & Dachs, 2002).
* Recursive_elimination.py: Compute a recursive elimination algorithm to derive variable importance from RFR and ANN models (Yang et al. 2020)
* bin2D.py: simple function to bin 2D matrices, requiring the Pandas library 
* interp2D.py: wrapper function around scipy's "RBF" and "griddata" interpolation functions, looped to iterate through the months of input DMS observational data
* regrid_climatology.py: helper function calling bin2D.py and interp2D.py internally to upscale published DMS barnes interplolated climatologies, for comparison with 20 km resolution RFR and ANN predictions 
* NSIDC_bin_sea_ice.py: code to convert raw datafiles NSIDC sea ice concentrations from polar sterographic to geodesic projections, and bin the data for later interpolation.

#### PLEASE NOTE: The "taylorDiagram.py" script includes functions to generate a Taylor Diagram (Taylor, 2001). It is from the public domain and is NOT my creation - all credit goes to Yannick Copin (https://gist.github.com/ycopin/3342888). However, this version is included for compatability and is modified to do the following:

* includes functionality that enables the user to normalize the standard deviations (and RMSE contours) to the reference data (i.e. the reference point is at a standard deviation of 1.0). This also replaces the x-axis label with "Standard deviation (norm.)" when enabled.
* includes functionality that enables a legend and text boxes to be added to the figure.
* changes the correlation tick values/locations.
* changes the reference point to a gold star with increased size.
