# To add a new cell, type ''
# To add a new markdown cell, type ' [markdown]'

import pandas as pd 
import numpy as np
from sklearn import metrics
from sklearn.datasets import fetch_california_housing



# Fetching a regression dataset
data = fetch_california_housing()

# Pulling data and columns to create training dataframe
X = data['data']
colnames = data['feature_names']
y = data['target']

# Creating pandas dataframe 
df_feat = pd.DataFrame(X, columns=colnames)
df_feat.head()

df_target = pd.DataFrame(y)
df_target.rename(columns={0:'target'},inplace=True)
df_target.head()



from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_percentage_error as mape
def eval_metrics(X,y,model):
    stat_model = {
        'rf': RandomForestRegressor(n_jobs=-1),
        'dt' : DecisionTreeRegressor()
    }
    
    X = X.to_numpy().reshape(-1,1)
    y = y.to_numpy().reshape(-1,1).ravel()
    #breakpoint()
    estimator = stat_model[model].fit(X,y)
    predictions = estimator.predict(y.reshape(-1,1))
    auc = mape(y,predictions)
    return auc



from sklearn.metrics import roc_auc_score
def greed_fs(n_features,X,y,model):
    best_scores=[]
    best_features=[]
    best_score = 0
    best_feature = ''
    for column in X.columns:
       curr_score = eval_metrics(X[column],y,model)
       if(curr_score>best_score):
           best_score = curr_score
           best_feature = column
           best_scores.append(best_score)
           best_features.append(best_feature)

    best_features.sort(reverse=True)
    best_scores.sort(reverse=True)
    print('Best feature for modeling is ',best_features[0:n_features])
    print('Socres for the feature is ',best_scores[0:n_features])


greed_fs(3,df_feat,df_target,'rf')






