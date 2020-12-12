import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import sklearn
import seaborn as sns
import xlrd
import time
import statsmodels.api as sm
import itertools


def CarloCrecco():
    print("Cucinaaa!")


def processSubset(y, X, feature_set, weights):
    import time
    # Fit model on feature_set and calculate RSS
    model = sm.WLS(y, X[list(feature_set)], weigths=1. / (weights ** 2))
    regr = model.fit()
    Y_pred = regr.predict(X[list(feature_set)])
    RSS = ((Y_pred - y) ** 2).sum()
    number_of_predictors = len(feature_set)
    return {"model": regr, "RSS": RSS, "number_of_predictors": number_of_predictors,
            "name_of_predictors": list(feature_set), "Y_pred": Y_pred}


def forward(y, X, predictors, weights, yesPrint):
    # Pull out predictors we still need to process
    remaining_predictors = [p for p in X.columns if p not in predictors]

    tic = time.time()

    results = []

    for p in remaining_predictors:
        results.append(processSubset(y, X, predictors + [p], weights))

    # Wrap everything up in a nice dataframe
    models = pd.DataFrame(results)

    # Choose the model with the highest RSS
    best_model = models.loc[models['RSS'].argmin()]

    toc = time.time()
    if yesPrint:
        display(models)
        print("Processed ", models.shape[0], "models on", len(predictors) + 1, "predictors in", (toc - tic), "seconds.")

    # Return the best model, along with some other useful information about the model
    return best_model


def backward(Y, X, predictors, weights, yesPrint):
    tic = time.time()

    results = []

    # We do so, because the next for-cycle will not compute the model with the max
    # results.append(fn.processSubset(Y, X, pred, weights))   

    for combo in itertools.combinations(predictors, len(predictors) - 1):
        results.append(processSubset(Y, X, combo, weights))

    # Wrap everything up in a nice dataframe
    models = pd.DataFrame(results)

    # Choose the model with the highest RSS
    best_model = models.loc[models['RSS'].argmin()]

    toc = time.time()
    if yesPrint:
        display(models)
        print("Processed ", models.shape[0], "models on", len(predictors) - 1, "predictors in", (toc - tic), "seconds.")

    # Return the best model, along with some other useful information about the model
    return best_model


# Funzione per calcolare
def mainForward(X, Y, weights, yesPrint=False):
    models_fwd = pd.DataFrame(columns=["RSS", "model", "number_of_predictors", "name_of_predictors", "Y_pred"])

    tic = time.time()
    predictors = []

    for i in range(1, len(X.columns) + 1):
        models_fwd.loc[i] = forward(Y, X, predictors, weights, yesPrint)
        predictors = models_fwd.loc[i]["model"].model.exog_names

    toc = time.time()
    print("Total elapsed time:", (toc - tic), "seconds.")

    return models_fwd


def mainBackward(X, Y, weights, yesPrint=False):
    models_bwd = pd.DataFrame(columns=["RSS", "model", "number_of_predictors", "name_of_predictors", "Y_pred"])

    tic = time.time()
    predictors = X.columns

    while (len(predictors) > 1):
        offset = len(predictors) - 1
        models_bwd.loc[offset] = backward(Y, X, predictors, weights, yesPrint)
        predictors = models_bwd.loc[offset]["model"].model.exog_names

    toc = time.time()
    print("Total elapsed time:", (toc - tic), "seconds.")

    return models_bwd


def compute_criteria(group_of_models):
    for i in range(1, group_of_models.shape[0] + 1):
        model = group_of_models.loc[i, "model"]
        group_of_models.loc[i, "aic"] = model.aic
        group_of_models.loc[i, "bic"] = model.bic
        group_of_models.loc[i, "mse"] = model.mse_total
        group_of_models.loc[i, "adj_rsquare"] = model.rsquared_adj
    return group_of_models


# GRAFICIC

def plot_response_over_prediction(response, prediction, title="Graph", figsize=(12, 8), wrapper=None):
    df = pd.DataFrame({'response': response, 'prediction': prediction, "diff": abs(response - prediction)})
    ax1 = df.reset_index().plot(kind='scatter', x='index', y='response', color='r', figsize=figsize, title=title)
    ax2 = df.reset_index().plot(x='index', y='prediction', color='b', ax=ax1, figsize=figsize)
    ax3 = df.reset_index().plot(x='index', y='diff', color='g', ax=ax1, figsize=figsize)
    if wrapper is not None:
        model = wrapper.model
        X = model.exog
        dt = wrapper.get_prediction(X).summary_frame(alpha=0.05)
        # y_prd = dt['mean']
        # yprd_ci_lower = dt['obs_ci_lower']
        # yprd_ci_upper = dt['obs_ci_upper']
        ym_ci_lower = dt['mean_ci_lower']
        ym_ci_upper = dt['mean_ci_upper']
        _ = plt.plot(np.linspace(start=1, stop=len(ym_ci_lower), num=len(ym_ci_lower)), ym_ci_lower, color="darkgreen",
                     linestyle="--",
                     label="Confidence Interval", ax=ax1)
        _ = plt.plot(np.linspace(start=1, stop=len(ym_ci_lower), num=len(ym_ci_lower)), ym_ci_upper, color="darkgreen",
                     linestyle="--", ax=ax1)
        _ = plt.legend()
        plt.show()
    return;


def selectBestForEachCriteria(models_fwd, criteriaToMin, criteriaToMax, toPrint=False):
    dict_results_best_models = {}

    for criteria in criteriaToMin:
        row = models_fwd.loc[models_fwd[criteria].argmin()]
        modelFeatures = row["model"].model.exog_names
        if "intercept" not in modelFeatures:
            modelFeatures.append("intercept")
        criteriaValue = row[criteria]
        degressOfFreedom = row["model"].model.df_model
        if toPrint:
            print("The criteria is: " + criteria)
            print("Features: " + str(modelFeatures))
            print("Criteria value: " + str(criteriaValue))
            print("Degrees of freedom: " + str(degressOfFreedom + 1))
            print()
        dict_results_best_models[criteria] = row

    for criteria in criteriaToMax:
        row = models_fwd.loc[models_fwd[criteria].argmax()]
        modelFeatures = row["model"].model.exog_names
        if "intercept" not in modelFeatures:
            modelFeatures.append("intercept")
        criteriaValue = row[criteria]
        degressOfFreedom = row["model"].model.df_model
        if toPrint:
            print("The criteria is: " + criteria)
            print("Features: " + str(modelFeatures))
            print("Criteria value: " + str(criteriaValue))
            print("Degrees of freedom: " + str(degressOfFreedom + 1))
            print()
        dict_results_best_models[criteria] = row

    best_models = pd.DataFrame(dict_results_best_models).T

    return best_models
