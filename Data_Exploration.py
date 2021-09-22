# Kaggle House Prices Data Exploration
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

df_train = pd.read_csv('./data/train.csv')

#print(df_train.columns)

#plt.scatter(df_train['SalePrice'], df_train['BsmtFinType1'])  

#print(df_train['SalePrice'].describe())

#sns.distplot(df_train['SalePrice'])

#print("Skewness: %f" % df_train['SalePrice'].skew())
#print("Kurtosis: %f" % df_train['SalePrice'].kurt())

#var = 'GrLivArea'
#data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
#data.plot.scatter(x=var, y='SalePrice')

#var = 'TotalBsmtSF'
#data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
#data.plot.scatter(x=var, y='SalePrice')

corrmat = df_train.corr()

#f, ax = plt.subplots(figsize=(12, 9))

#sns.heatmap(corrmat, vmax=.8, square=True)

#k = 10 # num vars for heatmap
#cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
#cm = np.corrcoef(df_train[cols].values.T)
#sns.set(font_scale=1.25)
#hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f',
#                 annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

#sns.set()

#cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF',
#        'FullBath', 'YearBuilt']

#sns.pairplot(df_train[cols], size=2.5)

total = df_train.isnull().sum().sort_values(ascending=False)
percent = (100.*(df_train.isnull().sum()/df_train.isnull().count())).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
#print(missing_data.head(20))

df_train = df_train.drop((missing_data[missing_data['Total'] > 1]).index, 1)
df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)
#print(df_train.isnull().sum().max())

saleprice_scaled = StandardScaler().fit_transform(df_train['SalePrice'][:,np.newaxis])
low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]
high_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]

print('outer (low) range of dist:')
print(low_range)
print('\nouter (high) range of dist:')
print(high_range)

var = 'GrLivArea'
#data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
#data.plot.scatter(x=var, y='SalePrice')

df_train.sort_values(by=var, ascending=False)[:2]
df_train = df_train.drop(df_train[df_train['Id'] ==1299].index)
df_train = df_train.drop(df_train[df_train['Id'] ==524].index)

var = 'TotalBsmtSF'
#data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
#data.plot.scatter(x=var, y='SalePrice')

df_train['SalePrice'] = np.log(df_train['SalePrice'])
df_train['GrLivArea'] = np.log(df_train['GrLivArea'])

df_train['HasBsmt'] = pd.Series(len(df_train['TotalBsmtSF']), index=df_train.index)
df_train['HasBsmt'] = 0
df_train.loc[df_train['TotalBsmtSF']>0, 'HasBsmt'] = 1
df_train.loc[df_train['HasBsmt']==1, 'TotalBsmtSF'] = np.log(df_train['TotalBsmtSF'])

sns.distplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], fit=norm)
#fig = plt.figure()
#res = stats.probplot

plt.show()
