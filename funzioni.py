import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import sklearn
import seaborn as sns
import xlrd
import time
import statsmodels.api as sm

def CarloCrecco():
    print("Cuindknpjbjfenjja!")
    
def processSubset(y, X, feature_set, weights):
    import time
    # Fit model on feature_set and calculate RSS
    #model = sm.OLS(y,X[list(feature_set)])
    model = sm.WLS(y,X[list(feature_set)], weigths = 1. /weights ** 2)
    #model = sm.GLS(y,X[list(feature_set)], sigma = weights**2.0)
    regr = model.fit()
    Y_pred=regr.predict(X[list(feature_set)])
    RSS = ((Y_pred - y) ** 2).sum()
    number_of_predictors = len(feature_set)
    return {"model":regr, "RSS":RSS, "number_of_predictors": number_of_predictors, "name_of_predictors": list(feature_set),"Y_pred":Y_pred}

def forward(y, X, predictors, weights, yesPrint):

    # Pull out predictors we still need to process
    remaining_predictors = [p for p in X.columns if p not in predictors]
    
    tic = time.time()
    
    results = []
    
    for p in remaining_predictors:
        results.append(processSubset(y,X,predictors+[p],weights))
    
    # Wrap everything up in a nice dataframe
    models = pd.DataFrame(results)
    if yesPrint:
        display(models)
    
    # Choose the model with the highest RSS
    best_model = models.loc[models['RSS'].argmin()]
    
    toc = time.time()
    print("Processed ", models.shape[0], "models on", len(predictors)+1, "predictors in", (toc-tic), "seconds.")
    
    # Return the best model, along with some other useful information about the model
    return best_model

# Funzione per calcolare 
def mainForward(X, Y, weights, yesPrint = False):
    models_fwd = pd.DataFrame(columns=["RSS", "model", "number_of_predictors", "name_of_predictors","Y_pred"])

    tic = time.time()
    predictors = []

    for i in range(1,len(X.columns)+1):    
        models_fwd.loc[i] = forward(Y, X, predictors, weights, yesPrint)
        predictors = models_fwd.loc[i]["model"].model.exog_names

    toc = time.time()
    print("Total elapsed time:", (toc-tic), "seconds.")
    return models_fwd

def compute_criteria(group_of_models):
    for i in range(1, group_of_models.shape[0]+1):
        model = group_of_models.loc[i,"model"]
        group_of_models.loc[i,"aic"] = model.aic
        group_of_models.loc[i,"bic"] = model.bic
        group_of_models.loc[i,"mse"] = model.mse_total
        group_of_models.loc[i,"adj_rsquare"] = model.rsquared_adj
    return group_of_models