# Code for Kaggle ML House Prices Competition
# scikit-learn version

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import sklearn.linear_model as linear_model
import sklearn.neural_network as neural_network
import sklearn.tree as tree
import sklearn.svm as svm
import seaborn as sns
import xgboost as xgb
import sklearn.preprocessing as prp
import sys
np.set_printoptions(threshold=sys.maxsize)

# Import Data
train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

# Train Data
quantitative = [f for f in train.columns if train.dtypes[f] != 'object']
quantitative.remove('SalePrice')
quantitative.remove('Id')
qualitative = [f for f in train.columns if train.dtypes[f] == 'object']

quantitative2 = [f for f in test.columns if test.dtypes[f] != 'object']
quantitative2.remove('Id')
qualitative2 = [f for f in test.columns if test.dtypes[f] == 'object']

missing = train.isnull().sum()
missing = missing[missing > 0]
missing.sort_values(inplace=True)

missing2 = test.isnull().sum()
missing2 = missing2[missing2 > 0]
missing2.sort_values(inplace=True)

def encode(frame, feature, test):
    ordering = pd.DataFrame()
    ordering['val'] = frame[feature].unique()
    ordering.index = ordering.val

    if test == False:
        ordering['spmean'] = frame[[feature,'SalePrice']].groupby(feature).mean()['SalePrice']
        ordering = ordering.sort_values('spmean')
    ordering['ordering'] = range(1, ordering.shape[0]+1)
    ordering = ordering['ordering'].to_dict()

    for cat, o in ordering.items():
        frame.loc[frame[feature] == cat, feature+'_E'] = o

# Error function
def error(actual, predicted):
    actual = np.log(actual)
    predicted = np.log(predicted)
    return np.sqrt(np.sum(np.square(actual-predicted)/len(actual)))

# Log transform
def log_transform(feature):
    train[feature] = np.log1p(train[feature].values)
    test[feature] = np.log1p(test[feature].values)

def quadratic(feature):
    train[feature+'2'] = train[feature]**2
    test[feature+'2'] = test[feature]**2

# Train Data
qual_encoded = []
for q in qualitative:
    encode(train, q, False)
    qual_encoded.append(q+'_E')

# Test Data
qual_encoded2 = []
for q in qualitative:
    encode(test, q, True)
    qual_encoded2.append(q+'_E')

log_transform('GrLivArea')
log_transform('1stFlrSF')
log_transform('2ndFlrSF')
log_transform('TotalBsmtSF')
log_transform('LotArea')
log_transform('LotFrontage')
log_transform('KitchenAbvGr')
log_transform('GarageArea')

quadratic('OverallQual')
quadratic('YearBuilt')
quadratic('YearRemodAdd')
quadratic('TotalBsmtSF')
quadratic('2ndFlrSF')
quadratic('Neighborhood_E')
quadratic('RoofMatl_E')
quadratic('GrLivArea')

qdr = ['OverallQual2', 'YearBuilt2', 'YearRemodAdd2', 'TotalBsmtSF2',
        '2ndFlrSF2', 'Neighborhood_E2', 'RoofMatl_E2', 'GrLivArea2']

# Boolean Terms
train['HasBasement'] = train['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
train['HasGarage'] = train['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
train['Has2ndFloor'] = train['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
train['HasMasVnr'] = train['MasVnrArea'].apply(lambda x: 1 if x > 0 else 0)
train['HasWoodDeck'] = train['WoodDeckSF'].apply(lambda x: 1 if x > 0 else 0)
train['HasPorch'] = train['OpenPorchSF'].apply(lambda x: 1 if x > 0 else 0)
train['HasPool'] = train['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
train['IsNew'] = train['YearBuilt'].apply(lambda x: 1 if x > 2000 else 0)

# Test data
test['HasBasement'] = test['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
test['HasGarage'] = test['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
test['Has2ndFloor'] = test['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
test['HasMasVnr'] = test['MasVnrArea'].apply(lambda x: 1 if x > 0 else 0)
test['HasWoodDeck'] = test['WoodDeckSF'].apply(lambda x: 1 if x > 0 else 0)
test['HasPorch'] = test['OpenPorchSF'].apply(lambda x: 1 if x > 0 else 0)
test['HasPool'] = test['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
test['IsNew'] = test['YearBuilt'].apply(lambda x: 1 if x > 2000 else 0)

boolean = ['HasBasement', 'HasGarage', 'Has2ndFloor', 'HasMasVnr', 'HasWoodDeck',
            'HasPorch', 'HasPool', 'IsNew']

features = quantitative + qual_encoded + boolean + qdr
features2 = quantitative2 + qual_encoded2 + boolean + qdr

model = linear_model.LinearRegression()

X = train[features].fillna(0).values
X_test = test[features2].fillna(0).values
Y = train['SalePrice'].values 
model.fit(X, np.log(Y))
Ypred = np.exp(model.predict(X))

print("Train Error: {:.5f}\n".format(error(Y, Ypred)))

print("\n\nNow xgBoost it...")

dtrain = xgb.DMatrix(X, label = Y)
dtest = xgb.DMatrix(X_test)

params = {"max_depth":2, "eta":0.1}
model = xgb.cv(params, dtrain,  num_boost_round=500, early_stopping_rounds=100)

#model.loc[30:,["test-rmse-mean", "train-rmse-mean"]].plot()

model_xgb = xgb.XGBRegressor(n_estimators=360, max_depth=20, learning_rate=0.1) #the params were tuned using xgb.cv
model_xgb.fit(X, np.log(Y))
Ypred2 = np.exp(model_xgb.predict(X))

print("XGBoost Train Error: {:.5f}\n".format(error(Y, Ypred2)))

Ypred3 = np.exp(model_xgb.predict(X_test))
#Ypred2 = np.exp(model.predict(X_test))

output = pd.DataFrame({'Id': test.get('Id').values, 'SalePrice': Ypred3})

output.to_csv('out.csv', index=False)

