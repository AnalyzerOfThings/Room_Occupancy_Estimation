import pandas as pd
import model_functions
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('Occupancy_Estimation.csv')
data['Date'] = pd.to_datetime(data['Date'], format='%Y/%m/%d')
data['Date'] = [_.strftime('%Y-%m-%d') for _ in data['Date']]
data['Time'] = pd.to_datetime(data['Time'])
data['Time'] = [_.strftime('%H:%M:%S') for _ in data['Time']]
data['Date'][data['Date'] == '2017-12-25'] = '2017-12-24'
clubbed_data = data.copy()
clubbed_data['Room_Occupancy_Count'].loc[clubbed_data
                                         ['Room_Occupancy_Count']==3] = 2

a = data.columns[2:16]
for col in a:
    data[col] = StandardScaler().fit_transform(data[col][:,np.newaxis])

features = ['S6_PIR', 'S7_PIR']
x = clubbed_data[features]
y = clubbed_data['Room_Occupancy_Count']
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)

# Logistic Regression using PIR

logreg = LogisticRegression()
logreg.fit(x_train, y_train)
preds_train = logreg.predict(x_train)
preds_test = logreg.predict(x_test)

sns.lineplot(x=y_train, y=preds_train)
plt.xlabel('Actual')
plt.ylabel('Predictions')
plt.title('P6, P7 Model Training')
plt.show()

sns.lineplot(x=y_test, y=preds_test)
plt.xlabel('Actual')
plt.ylabel('Predictions')
plt.title('P6, P7 Model Testing')
plt.show()

for col in features:
    sns.scatterplot(x_train[col], label = col)
    sns.scatterplot(preds_train, label = 'Predictions')
    plt.title('P6, P7 Model Training')
    plt.ylabel('Values')
    plt.show()
    
for col in features:
    sns.scatterplot(x_test[col], label = col)
    sns.scatterplot(preds_test, label = 'Predictions')
    plt.title('P6, P7 Model Testing')
    plt.ylabel('Values')
    plt.show()
    
print('Accuracy on training data: ', model_functions.accuracy(y_train,
                                                    pd.Series(preds_train)))
print('Accuracy on testing data: ', model_functions.accuracy(y_test,
                                                    pd.Series(preds_test)))

# Classification using rules on S6_PIR & S7_PIR

# Rule 1 ==> S6_PIR = 0 -> 0 people
# Rule 2 ==> S6_PIR = 0 and S7_PIR = 1 -> 1 person
# Rule 3 ==> S6_PIR = 1 and S7_PIR = 1 -> 2 people

preds_train = []
preds_test = []

for i in range(len(x_train)):
    if x_train['S6_PIR'].iloc[i] == 0 and x_train['S7_PIR'].iloc[i] == 0:
        preds_train.append(0)
    elif x_train['S6_PIR'].iloc[i] == 0 and x_train['S7_PIR'].iloc[i] == 1:
        preds_train.append(1)
    elif x_train['S6_PIR'].iloc[i] == 1 and x_train['S7_PIR'].iloc[i] == 0:
        preds_train.append(1)
    else:
        preds_train.append(2)
        
for i in range(len(x_test)):
    if x_test['S6_PIR'].iloc[i] == 0 and x_test['S7_PIR'].iloc[i] == 0:
        preds_test.append(0)
    elif x_test['S6_PIR'].iloc[i] == 0 and x_test['S7_PIR'].iloc[i] == 1:
        preds_test.append(1)
    elif x_test['S6_PIR'].iloc[i] == 1 and x_test['S7_PIR'].iloc[i] == 0:
        preds_test.append(1)
    else:
        preds_test.append(2)
    
    

print('Accuracy on training data: ', model_functions.accuracy(y_train,
                                                    pd.Series(preds_train)))
print('Accuracy on testing data: ', model_functions.accuracy(y_test,
                                                    pd.Series(preds_test)))

for col in features:
    sns.scatterplot(x_train[col], label = col)
    sns.scatterplot(preds_train, label = 'Predictions')
    plt.title('P6, P7 Rules Training')
    plt.ylabel('Values')
    plt.show()
    
for col in features:
    sns.scatterplot(x_test[col], label = col)
    sns.scatterplot(preds_test, label = 'Predictions')
    plt.title('P6, P7 Rules Testing')
    plt.ylabel('Values')
    plt.show()


sns.lineplot(x=y_train, y=preds_train)
plt.xlabel('Actual')
plt.ylabel('Predictions')
plt.title('P6, P7 Rules Training')
plt.show()

sns.lineplot(x=y_test, y=preds_test)
plt.xlabel('Actual')
plt.ylabel('Predictions')
plt.title('P6, P7 Rules Testing')
plt.show()

#####################################################################
# Only 2 and 3 data

clubbed_data = data[data['Room_Occupancy_Count'].isin([2,3])] 
features = ['S1_Temp','S2_Temp','S3_Temp','S4_Temp','S1_Light','S2_Light',
            'S3_Light','S4_Light','S1_Sound','S2_Sound','S3_Sound',
            'S4_Sound','S5_CO2','S5_CO2_Slope','S6_PIR','S7_PIR']

x = clubbed_data[features]
y = clubbed_data['Room_Occupancy_Count']
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)


logreg = LogisticRegression()
logreg.fit(x_train, y_train)
preds_train = logreg.predict(x_train)
preds_test = logreg.predict(x_test)
print('Accuracy on training data: ', model_functions.accuracy(y_train,
                                                    pd.Series(preds_train)))
print('Accuracy on testing data: ', model_functions.accuracy(y_test,
                                                    pd.Series(preds_test)))
'''
x_train.reset_index()
preds_train = pd.DataFrame({'preds_train':preds_train})
preds_train['index'] = x_train.index
for col in features:
    sns.scatterplot(y = x_train[col], x = x_train.index, label = col)
    sns.scatterplot(y=preds_train['preds_train'], x = preds_train['index'],
                    label = 'Predictions')
    plt.title('2, 3 Model Training')
    plt.ylabel('Values')
    plt.show()


x_test.reset_index()
preds_test = pd.DataFrame({'preds_test':preds_test})
preds_test['index'] = x_test.index
for col in features:
    sns.scatterplot(y = x_test[col], x = x_test.index, label = col)
    sns.scatterplot(y=preds_test['preds_test'], x = preds_test['index'],
                    label = 'Predictions')
    plt.title('2, 3 Model Testing')
    plt.ylabel('Values')
    plt.show()
    
'''
#######################################################################

sns.lineplot(x=y_train, y=preds_train)
plt.xlabel('Actual')
plt.ylabel('Predictions')
plt.title('2,3 Model Training')
plt.show()

sns.lineplot(x=y_test, y=preds_test)
plt.xlabel('Actual')
plt.ylabel('Predictions')
plt.title('2,3 Model Testing')
plt.show()

preds_train = []
preds_test = []

for i in range(len(x_train)):
    if x_train['S4_Light'].iloc[i] > 0.5:
        if x_train['S5_CO2'].iloc[i] < 0.5:
            preds_train.append(2)
        else:
            preds_train.append(3)
    else:
        if x_train['S5_CO2'].iloc[i] > 0.5:
            preds_train.append(3)
        else:
            preds_train.append(2)
            
for i in range(len(x_test)):
    if x_test['S4_Light'].iloc[i] > 0.5:
        if x_test['S5_CO2'].iloc[i] < 0.5:
            preds_test.append(2)
        else:
            preds_test.append(3)
    else:
        if x_test['S5_CO2'].iloc[i] > 0.5:
            preds_test.append(3)
        else:
            preds_test.append(2)


print('Accuracy on training data: ', model_functions.accuracy(y_train,
                                                    pd.Series(preds_train)))
print('Accuracy on testing data: ', model_functions.accuracy(y_test,
                                                    pd.Series(preds_test)))

sns.lineplot(x=y_train, y=preds_train)
plt.xlabel('Actual')
plt.ylabel('Predictions')
plt.title('2, 3 Rules Training')
plt.show()

sns.lineplot(x=y_test, y=preds_test)
plt.xlabel('Actual')
plt.ylabel('Predictions')
plt.title('2, 3 Rules Testing')
plt.show()

features = ['S1_Temp','S2_Temp','S3_Temp','S4_Temp','S1_Light','S2_Light',
            'S3_Light','S4_Light','S1_Sound','S2_Sound','S3_Sound',
            'S4_Sound','S5_CO2','S5_CO2_Slope','S6_PIR','S7_PIR']

x = clubbed_data[features]
y = clubbed_data['Room_Occupancy_Count']
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)

dtree = DecisionTreeClassifier()
dtree.fit(x_train, y_train)
preds_train = dtree.predict(x_train)
preds_test = dtree.predict(x_test)

sns.lineplot(x=y_train, y=preds_train)
plt.xlabel('Actual')
plt.ylabel('Predictions')
plt.title('Decision Tree Training')
plt.show()

sns.lineplot(x=y_test, y=preds_test)
plt.xlabel('Actual')
plt.ylabel('Predictions')
plt.title('Decision Tree Testing')
plt.show()

print('Accuracy on training data: ', model_functions.accuracy(y_train,
                                                    pd.Series(preds_train)))
print('Accuracy on testing data: ', model_functions.accuracy(y_test,
                                                    pd.Series(preds_test)))

rf = RandomForestClassifier()
rf.fit(x_train, y_train)
preds_train = rf.predict(x_train)
preds_test = rf.predict(x_test)

print('Accuracy on training data: ', model_functions.accuracy(y_train,
                                                    pd.Series(preds_train)))
print('Accuracy on testing data: ', model_functions.accuracy(y_test,
                                                    pd.Series(preds_test)))

sns.lineplot(x=y_train, y=preds_train)
plt.xlabel('Actual')
plt.ylabel('Predictions')
plt.title('Random Forest Training')
plt.show()

sns.lineplot(x=y_test, y=preds_test)
plt.xlabel('Actual')
plt.ylabel('Predictions')
plt.title('Random Forest Testing')
plt.show()

abc = AdaBoostClassifier(n_estimators=50, learning_rate=1)
abc.fit(x_train, y_train)
preds_train = abc.predict(x_train)
preds_test = abc.predict(x_test)

print('Accuracy on training data: ', model_functions.accuracy(y_train,
                                                    pd.Series(preds_train)))
print('Accuracy on testing data: ', model_functions.accuracy(y_test,
                                                    pd.Series(preds_test)))

sns.lineplot(x=y_train, y=preds_train)
plt.xlabel('Actual')
plt.ylabel('Predictions')
plt.title('AdaBoost Training')
plt.show()

sns.lineplot(x=y_test, y=preds_test)
plt.xlabel('Actual')
plt.ylabel('Predictions')
plt.title('AdaBoost Testing')
plt.show()
