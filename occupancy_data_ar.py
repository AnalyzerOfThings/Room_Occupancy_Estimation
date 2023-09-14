import pandas as pd
import model_functions
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from statsmodels.tsa.stattools import adfuller, acf, pacf
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('Occupancy_Estimation.csv')
data['Date'] = pd.to_datetime(data['Date'], format='%Y/%m/%d')
data['Date'] = [_.strftime('%Y-%m-%d') for _ in data['Date']]
data['Time'] = pd.to_datetime(data['Time'])
data['Time'] = [_.strftime('%H:%M:%S') for _ in data['Time']]
data['Date'][data['Date'] == '2017-12-25'] = '2017-12-24'

s = plot_acf(data['Room_Occupancy_Count'])
plt.show(s) # acf decays slowly

s = plot_pacf(data['Room_Occupancy_Count'])
plt.show(s) # pacf ~ 0 after lag = 1 ==> AR(1)
print('ADF: ',adfuller(data['Room_Occupancy_Count'])[0]) # stationary time series
data['ROC'] = StandardScaler().fit_transform(
    data['Room_Occupancy_Count'][:,np.newaxis])
data['ROC_Shifted'] = data['ROC'].shift()
data = data.dropna()
x = data['ROC_Shifted']
y = data[['ROC', 'Room_Occupancy_Count']]
x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=0)
x_train = np.array(x_train).reshape(-1, 1)
x_test = np.array(x_test).reshape(-1, 1)

model = LinearRegression()
model.fit(x_train, y_train['ROC'])
preds = list(model.predict(x_test))

for i in range(len(preds)):
    preds[i] = round(preds[i])
    if preds[i]<0:
        preds[i]=0

sns.scatterplot(x = y_test['Room_Occupancy_Count'], y = preds)
plt.xlabel('Actual')
plt.ylabel('Preds')
plt.show()

print('Accuracy: ',model_functions.accuracy(y_test['Room_Occupancy_Count'], 
                               pd.Series(preds)))
print('Coef: ', float(model.coef_), 
      ' (If coef < 1, Time Series is STATIONARY)')