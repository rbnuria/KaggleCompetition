# -*- coding: utf-8 -*-

##### VERSIÓN 1:
## Seguimos basándonos en la versión 1, pues es la mejor conseguida hasta el momento pero realizamos sustituímos los valores perdidos
## missing values utilizando vecino más cercano (fichero R)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import ensemble, tree, linear_model
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.utils import shuffle

import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('notNA_train.csv')
test = pd.read_csv('notNA_test.csv')


np.random.seed(11223344)

# Prints R2 and RMSE scores
def get_score(prediction, lables):    
    print('R2: {}'.format(r2_score(prediction, lables)))
    print('RMSE: {}'.format(np.sqrt(mean_squared_error(prediction, lables))))
    print('RSMLE: {}'.format(np.sqrt(mean_squared_error(np.log(prediction), np.log(lables)))))

 

# Spliting to features and lables and deleting variable I don't need
train_labels = train.pop('SalePrice')

features = pd.concat([train, test], keys=['train', 'test'])


## Log transformation of labels
train_labels = np.log(train_labels)

#Quitamos
#features.drop(['OverallCond', 'BsmtFinSF2', 'KitchenAvgGr', 'EnclosedPorch', 'MiscVal', 'YrSold'], axis = 1, inpace = True)

#Las más importantes son -> OverallQual (con mucha diferencia), GrvArea, TotalBsmtSF, 'X1stFlrSF', 'GarageCars', 'GarageArea' -> Los normalizamos 

#Ponemos logaritmico los que no toman valor 0
features['OverallQual'] = np.log(features['OverallQual'])
features['GrLivArea'] = np.log(features['GrLivArea'])
features['X1stFlrSF'] = np.log(features['X1stFlrSF'])
features['MSSubClass'] = np.log(features['MSSubClass'])
features['LotFrontage'] = np.log(features['LotFrontage'])
features['LotArea'] = np.log(features['LotArea'])
features['OverallCond'] = np.log(features['OverallCond'])
features['YearBuilt'] = np.log(features['YearBuilt'])
features['YearRemodAdd'] = np.log(features['YearRemodAdd'])
features['X1stFlrSF'] = np.log(features['X1stFlrSF'])
features['YrSold'] = np.log(features['YrSold'])
features['MoSold'] = np.log(features['MoSold'])






## Standardizing numeric features
numeric_features = features.loc[:,['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt',
'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'X1stFlrSF',
'X2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr',
'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 
'EnclosedPorch', 'X3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']]


numeric_features_standardized = (numeric_features - numeric_features.mean())/numeric_features.std()



# Getting Dummies from Condition1 and Condition2
conditions = set([x for x in features['Condition1']] + [x for x in features['Condition2']])
dummies = pd.DataFrame(data=np.zeros((len(features.index), len(conditions))),
                       index=features.index, columns=conditions)
for i, cond in enumerate(zip(features['Condition1'], features['Condition2'])):
    dummies.ix[i, cond] = 1
features = pd.concat([features, dummies.add_prefix('Condition_')], axis=1)
features.drop(['Condition1', 'Condition2'], axis=1, inplace=True)

# Getting Dummies from Exterior1st and Exterior2nd
exteriors = set([x for x in features['Exterior1st']] + [x for x in features['Exterior2nd']])
dummies = pd.DataFrame(data=np.zeros((len(features.index), len(exteriors))),
                       index=features.index, columns=exteriors)
for i, ext in enumerate(zip(features['Exterior1st'], features['Exterior2nd'])):
    dummies.ix[i, ext] = 1
features = pd.concat([features, dummies.add_prefix('Exterior_')], axis=1)
features.drop(['Exterior1st', 'Exterior2nd'], axis=1, inplace=True)

# Getting Dummies from all other categorical vars
for col in features.dtypes[features.dtypes == 'object'].index:
    for_dummy = features.pop(col)
    features = pd.concat([features, pd.get_dummies(for_dummy, prefix=col)], axis=1)
    
### Copying features
features_standardized = features.copy()

### Replacing numeric features by standardized values
features_standardized.update(numeric_features_standardized)

### Splitting features
train_features = features.loc['train'].drop('Id', axis=1).select_dtypes(include=[np.number]).values
test_features = features.loc['test'].drop('Id', axis=1).select_dtypes(include=[np.number]).values

### Splitting standardized features
train_features_st = features_standardized.loc['train'].drop('Id', axis=1).select_dtypes(include=[np.number]).values
test_features_st = features_standardized.loc['test'].drop('Id', axis=1).select_dtypes(include=[np.number]).values

### Shuffling train sets
train_features_st, train_features, train_labels = shuffle(train_features_st, train_features, train_labels, random_state = 5)

### Splitting
x_train, x_test, y_train, y_test = train_test_split(train_features, train_labels, test_size=0.1, random_state=200)
x_train_st, x_test_st, y_train_st, y_test_st = train_test_split(train_features_st, train_labels, test_size=0.1, random_state=200)

'''
Elastic Net
'''
ENSTest = linear_model.ElasticNetCV(alphas=[0.0001, 0.0005, 0.001, 0.01, 0.1, 1, 10], l1_ratio=[.01, .1, .5, .9, .99], max_iter=5000).fit(x_train_st, y_train_st)


'''
Gradient Boosting
'''
GBest = ensemble.GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=3, max_features='sqrt',
                                               min_samples_leaf=15, min_samples_split=10, loss='huber')

# Retraining models
GB_model = GBest.fit(train_features, train_labels)
ENST_model = ENSTest.fit(train_features_st, train_labels)

## Getting our SalePrice estimation
Final_labels = (np.exp(GB_model.predict(test_features)) + np.exp(ENST_model.predict(test_features_st))) / 2
Final_labels_train = (np.exp(ENST_model.predict(train_features_st)) + np.exp(GB_model.predict(train_features))) / 2

get_score(Final_labels_train, np.exp(train_labels))

## Saving to CSV
#pd.DataFrame({'Id': test.Id, 'SalePrice': Final_labels}).to_csv('submission16.csv', index =False)