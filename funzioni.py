def CarloCrecco():
    print("Cuina!")
    
def processSubset(X,feature_set,weights):
    # Fit model on feature_set and calculate RSS
    #model = sm.OLS(y,X[list(feature_set)])
    model = sm.WLS(y,X[list(feature_set)], weigths = 1. /weights ** 2)
    #model = sm.GLS(y,X[list(feature_set)], sigma = weights**2.0)
    regr = model.fit()
    RSS = ((regr.predict(X[list(feature_set)]) - y) ** 2).sum()
    number_of_predictors = len(feature_set)
    return {"model":regr, "RSS":RSS, "number_of_predictors": number_of_predictors}

def forward(X,predictors,weights):

    # Pull out predictors we still need to process
    remaining_predictors = [p for p in X.columns if p not in predictors]
    
    tic = time.time()
    
    results = []
    
    for p in remaining_predictors:
        results.append(processSubset(X,predictors+[p],weights))
    
    # Wrap everything up in a nice dataframe
    models = pd.DataFrame(results)
    display(models)
    #print()
    
    # Choose the model with the highest RSS
    best_model = models.loc[models['RSS'].argmin()]
    
    toc = time.time()
    print("Processed ", models.shape[0], "models on", len(predictors)+1, "predictors in", (toc-tic), "seconds.")
    
    # Return the best model, along with some other useful information about the model
    return best_model