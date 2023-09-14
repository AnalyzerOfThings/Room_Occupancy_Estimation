import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import model_functions
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('Occupancy_Estimation.csv')
data['Date'] = pd.to_datetime(data['Date'], format='%Y/%m/%d')
data['Date'] = [_.strftime('%Y-%m-%d') for _ in data['Date']]
data['Time'] = pd.to_datetime(data['Time'])
data['Time'] = [_.strftime('%H:%M:%S') for _ in data['Time']]
data['Date'][data['Date'] == '2017-12-25'] = '2017-12-24'

'''
# Removing data in the night as Room_Occupancy_Count always 0 in the night.

data = data.loc[(data['Time'] < '20:00:00') &
                (data['Time'] > '07:00:00')].reset_index(drop=True)
# Removing data collected on 2018-01-10 upto 17:01:00 as light readings
# of all sensors were weird during this period.

indices=[]
for i in range(len(data)):
    if data.iloc[i].Date=='2018-01-10' and data.iloc[i].Time < '17:01:00':
        indices.append(i)
data = data.drop(index=indices).reset_index(drop=True)
'''

features = ['S1_Light', 'S2_Light', 'S3_Light', 'S1_Temp', 'S5_CO2']
model_functions.handle_outliers(data, features, 'replace')
x = data[features]
y = data['Room_Occupancy_Count']

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)
x_test, x_validate, y_test, y_validate = train_test_split(x_test, y_test,
                                train_size=0.5, random_state=1)
model = LinearRegression()
model.fit(x_train, y_train)

# Evaluation on training data

preds_train = model_functions.round_items(list(model.predict(x_train)))

count=0
for i in range(len(preds_train)):
    if preds_train[i]==y_train.iloc[i]:
        count+=1
print('Train Accuracy: ',count/len(preds_train))
print('Train Adj R2: ', model_functions.adj_r2(y_train,
                                              pd.Series(preds_train), 5))

# Evaluation on testing data

preds_test = model_functions.round_items(list(model.predict(x_test)))

count=0
for i in range(len(preds_test)):
    if preds_test[i]==y_test.iloc[i]:
        count+=1
print('Test Accuracy: ',count/len(preds_test))
print('Test Adj R2: ', model_functions.adj_r2(y_test,
                                              pd.Series(preds_test), 5))

# Evaluation on validation data

preds_validate = model_functions.round_items(list(model.predict(x_validate)))
count=0
for i in range(len(preds_validate)):
    if preds_validate[i]==y_validate.iloc[i]:
        count+=1
print('Validate Accuracy: ',count/len(preds_validate))
print('Validate Adj R2: ', model_functions.adj_r2(y_validate,
                                             pd.Series(preds_validate), 5))
x_train['Room_Occupancy_Count'] = y_train
x_train['preds'] = preds_train
x_test['Room_Occupancy_Count'] = y_test
x_test['preds'] = preds_test
x_validate['Room_Occupancy_Count'] = y_validate
x_validate['preds'] = preds_validate

sns.scatterplot(x_train['Room_Occupancy_Count'], label='Actual')
sns.scatterplot(x_train['preds'], label='Preds')
plt.title('Train')
plt.show()

sns.scatterplot(x_test['Room_Occupancy_Count'], label='Actual')
sns.scatterplot(x_test['preds'], label='Preds')
plt.title('Test')
plt.show()

sns.scatterplot(x_validate['Room_Occupancy_Count'], label='Actual')
sns.scatterplot(x_validate['preds'], label='Preds')
plt.title('Validate')
plt.show()