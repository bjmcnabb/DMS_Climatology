# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 19:00:45 2022

@author: bcamc
"""


def recursive_elim(RFR_model, input_data):
    from sklearn.base import clone
    from sklearn import metrics
    import numpy as np
    import pandas as pd
    from tqdm import tqdm
    
    # define dict to save importances
    X_train, y_train, X_test, y_test = input_data[0], input_data[1], input_data[2], input_data[3]
    importances = pd.DataFrame()
    
    # calculate reference metrics using the testing dataset & OOB preditions
    base_R2 = RFR_model.score(X_test.values,y_test.values)
    base_mse = metrics.mean_squared_error(y_test.values, RFR_model.predict(X_test.values))
    base_oob_R2 = RFR_model.oob_score_
    base_oob_mse = metrics.mean_squared_error(y_train.values, RFR_model.oob_prediction_)
    importance = np.nan
    
    # save base model metrics
    importances.at['base',['importance','R2','oob_R2','mse','oob_mse']] = [importance, base_R2, base_oob_R2, base_mse, base_oob_mse]
    
    # loop to test each predictor seperately
    for var_ in tqdm(X_train.columns):
        # clone the model to reset fit
        RFR_test_model = clone(RFR_model)
        # set the new max features (1-total number of features)
        RFR_test_model.max_features = np.min(X_train.drop(var_, axis=1).shape)
        # fit the test model
        RFR_test_model.fit(X_train.drop(var_, axis=1).values, y_train.values)
        # calculate MSE & R2 for the testing dataset & OOB preditions
        drop_R2 = RFR_test_model.score(X_test.drop(var_, axis=1).values,y_test.values)
        drop_mse = metrics.mean_squared_error(y_test.values, RFR_test_model.predict(X_test.drop(var_, axis=1).values))
        drop_oob_R2 = RFR_test_model.oob_score_
        drop_oob_mse = metrics.mean_squared_error(y_train.values, RFR_test_model.oob_prediction_)
        # calculate importance metric
        importance = np.mean([drop_mse - base_mse, drop_oob_mse - base_oob_mse])
        # save metrics
        importances.at[var_,['importance','R2','oob_R2','mse','oob_mse']] = [importance, drop_R2, drop_oob_R2, drop_mse, drop_oob_mse]
    
    importances = importances.sort_values(by='importance', ascending=False)
    return importances 

def get_stats(RFR_model, importances, input_data):
    from sklearn.base import clone
    from sklearn import metrics
    import numpy as np
    import pandas as pd
    from tqdm import tqdm
    X_train, y_train, X_test, y_test = input_data[0], input_data[1], input_data[2], input_data[3]
    features = importances.index[:-1][::-1]
    i = 0
    stats = pd.DataFrame()
    for var_ in tqdm(features):
        i += 1
        RFR_test_model = clone(RFR_model)
        RFR_test_model.max_features = features[:i].shape[0]
        RFR_test_model.fit(X_train.loc[:,(features[:i])].values, y_train.values)
        R2 = RFR_test_model.oob_score_
        rmse = np.sqrt(metrics.mean_squared_error(np.sinh(y_test).values, np.sinh(RFR_test_model.predict(X_test.loc[:,(features[:i])].values))))
        stats.at[var_, ['R2','RMSE']] = [R2, rmse]
    return stats


