Scripts here provide the following:
*NN_model_frameworks.py: includes a PyTorch class that builds an artificial neural network for regression. Subfunctions include functionality to convert numpy arrays to torch datasets for compatability and to porduce a GIF of the training process (i.e. plots boht model fit and loss for each training epoch)  
*SO_mapping_templates.py: boilerplate to produce orthographic and plate carree projections of the Southern Ocean (below 40oS), with formatting to plot the location of relevant glaciers and ice shelves. Subfunctions include functionality for plotting contours (filled or unfilled), pcolormesh, or scatterplots.
*fluxes: subfunction for computing sea-air fluxes of DMS, using either the GM12 or SD02 parameterizations (see the associated manuscript).
*taylorDiagram.py
