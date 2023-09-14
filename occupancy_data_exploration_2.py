import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


data = pd.read_csv('Occupancy_Estimation.csv')
data['Date'] = pd.to_datetime(data['Date'], format='%Y/%m/%d')
data['Date'] = [_.strftime('%Y-%m-%d') for _ in data['Date']]
data['Time'] = pd.to_datetime(data['Time'])
data['Time'] = [_.strftime('%H:%M:%S') for _ in data['Time']]
data['Date'][data['Date'] == '2017-12-25'] = '2017-12-24'

data = data[data['Room_Occupancy_Count'].isin([2,3])]

for col in data.columns[2:16]:
    data[col] =  StandardScaler().fit_transform(data[col][:,np.newaxis])

# Make note of all vars, their range and their dtypes

print(data.dtypes)
for col in list(data.columns):
    print(col,': (min, max) =  ', (min(data[col]),max(data[col])))
    
# Analyse target

target = 'Room_Occupancy_Count'
print(data[target].describe())
sns.displot(data[target])
plt.show()
print('Skewness: ', data[target].skew())
print('Kurtosis: ',data[target].kurt())

contn_vars = ['S1_Temp', 'S2_Temp', 'S3_Temp', 'S4_Temp', 'S1_Light',
       'S2_Light', 'S3_Light', 'S4_Light', 'S1_Sound', 'S2_Sound', 'S3_Sound',
       'S4_Sound', 'S5_CO2', 'S5_CO2_Slope']


for var in contn_vars:
    ax = sns.scatterplot(data, x = var, y = target)
    plt.show()    
    
categ_vars = ['S6_PIR', 'S7_PIR']
for var in categ_vars:
    df = pd.concat([data[var], data[var]], axis=1)
    f, ax = plt.subplots(figsize = (16,8))
    fig = sns.boxplot(x = var, y = target, data = data)
    plt.show()

# Analyse features

corrmat = data.corr()
f, ax = plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=0.8, square=True)
plt.show()

k = 10
cols = corrmat.nlargest(k, target)[target].index
cm = np.corrcoef(data[cols].values.T)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f',
                 annot_kws={'size':10}, yticklabels=cols.values,
                 xticklabels=cols.values)
plt.show()

for col1 in cols:
    for col2 in cols:
        if col1!=col2:
            sns.scatterplot(data=data, x=col1, y=col2)
            plt.show()

# Missing Data

total = data.isnull().sum().sort_values(ascending=False)
percent = (data.isnull().sum()/data.isnull().count()).sort_values(ascending
                                                                  = False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)

''' Remove col if percent > 15 '''

# Outliers

target_scaled = StandardScaler().fit_transform(data[target][:,np.newaxis])
lowrange = target_scaled[target_scaled[:,0].argsort()][:10]
highrange = target_scaled[target_scaled[:,0].argsort()][-10:]
print('Lower end of distribution: ')
print(lowrange)
print('Higher end of distribution: ')
print(highrange)

# Check if vars conform to statistical test assumptions

## Normality

for col in cols:
    sns.distplot(data[col], fit=norm)
    fig = plt.figure()
    res = stats.probplot(data[col], plot=plt)
    plt.title(col+' Probability Plot')
    plt.show()

for col in cols:
    sns.scatterplot(data=data, y=target, x=col)
    plt.show()

# Enter dummy vars where necessary

#data = pd.get_dummies(data, columns=['Room_Occupancy_Count'])