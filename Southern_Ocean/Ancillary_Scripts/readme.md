![PyTorch_ANN_Fit](https://user-images.githubusercontent.com/68400556/161632855-8fa55e2e-be69-47d4-94a2-f52e9ad1a0eb.gif)

Scripts here provide the following:
* NN_model_frameworks.py: includes a PyTorch class that builds an artificial neural network for regression. Subfunctions include functionality to convert numpy arrays to torch datasets for compatability and to produce a GIF of the training process (i.e. plots both model fit and loss for each training epoch - see above)  
* SO_mapping_templates.py: boilerplate to produce orthographic and plate carree projections of the Southern Ocean (below 40oS), with formatting to plot the location of relevant glaciers and ice shelves. The subfunctions include functionality for plotting contours (filled or unfilled), pcolormesh, or scatterplots.
* Fluxes.py: function for computing sea-air fluxes of DMS, using either the GM12 or SD02 parameterizations (see the associated manuscript).

#### PLEASE NOTE: The "taylorDiagram.py" script includes functions to generate a Taylor Diagram (Taylor, 2001). It is from the public domain and is NOT my creation - all credit goes to Yannick Copin (https://gist.github.com/ycopin/3342888). However, this version is included for compatability and is modified to do the following:

* includes functionality that enables the user to normalize the standard deviations (and RMSE contours) to the reference data (i.e. the reference point is at a standard deviation of 1.0). This also replaces the x-axis label with "Standard deviation (norm.)" when enabled.
* includes functionality that enables a legend and text boxes to be added to the figure.
* changes the correlation tick values/locations.
* changes the reference point to a gold star and increases it's size.
