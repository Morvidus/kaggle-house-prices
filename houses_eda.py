# Kaggle House Prices EDA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import sklearn.linear_model as linear_model
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import KFold
from IPython.display import HTML, display
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

pd.options.display.max_rows = 1000
pd.options.display.max_columns = 20

train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

quantitative = [f for f in train.columns if train.dtypes[f] != 'object']
quantitative.remove('SalePrice')
quantitative.remove('Id')
qualitative = [f for f in train.columns if train.dtypes[f] == 'object']

missing = train.isnull().sum()

missing = missing[missing > 0]
missing.sort_values(inplace=True)
#missing.plot.bar()


##test_normality = lambda x: stats.shapiro(x.fillna(0))[1] < 0.01
##normal = pd.DataFrame(train[quantitative])
##normal = normal.apply(test_normality)
##print(not normal.any())

#f = pd.melt(train, value_vars=quantitative)
#g =sns.FacetGrid(f, col='variable', col_wrap=2, sharex=False, sharey=False)
#g = g.map(sns.distplot, 'value')

#for c in qualitative:
#    train[c] = train[c].astype('category')
#    if train[c].isnull().any():
#        train[c] = train[c].cat.add_categories(['MISSING'])
#        train[c] = train[c].fillna('MISSING')

#def boxplot(x, y, **kwargs):
#    sns.boxplot(x=x,y=y)
#    x=plt.xticks(rotation=90)

#f = pd.melt(train, id_vars=['SalePrice'], value_vars=qualitative)
#g = sns.FacetGrid(f, col='variable', col_wrap=2, sharex=False, sharey=False, size=5)
#g = g.map(boxplot, 'value', 'SalePrice')

def anova(frame):
anv = pd.DataFrame()
anv['feature'] = qualitative

pvals = []

for c in qualitative:
    samples = []
    for cls in frame[c].unique():
        s = frame[frame[c] == cls]['SalePrice'].values
        samples.append(s)
    pval = stats.f_oneway(*samples)[1]
    pvals.append(pval)
anv['pval'] = pvals
return anv.sort_values('pval')

a = anova(train)
a['disparity'] = np.log(1./a['pval'].values)
sns.barplot(data=a, x='feature', y='disparity')
x=plt.xticks(rotation=90)

def encode(frame, feature):
ordering = pd.DataFrame()
ordering['val'] = frame[feature].unique()
ordering.index = ordering.val
ordering['spmean'] = frame[[feature,'SalePrice']].groupby(feature).mean()['SalePrice']
ordering = ordering.sort_values('spmean')
ordering['ordering'] = range(1, ordering.shape[0]+1)
ordering = ordering['ordering'].to_dict()

for cat, o in ordering.items():
    frame.loc[frame[feature] == cat, feature+'_E'] = o

qual_encoded = []
for q in qualitative:
encode(train, q)
qual_encoded.append(q+'_E')

#print(qual_encoded)

def spearman(frame, features):
spr = pd.DataFrame()
spr['feature'] = features
spr['spearman'] = [frame[f].corr(frame['SalePrice'], 'spearman') for f in features]

spr = spr.sort_values('spearman')
plt.figure(figsize=(6, 0.25*len(features)))
sns.barplot(data=spr, y=features, x='spearman', orient='h')

features = quantitative + qual_encoded
spearman(train, features)
    
plt.show()
