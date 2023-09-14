import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import model_functions

data = pd.read_csv('Occupancy_Estimation.csv')
data['Date'] = pd.to_datetime(data['Date'], format='%Y/%m/%d')
data['Date'] = [_.strftime('%Y-%m-%d') for _ in data['Date']]
data['Time'] = pd.to_datetime(data['Time'])
data['Time'] = [_.strftime('%H:%M:%S') for _ in data['Time']]
data['Date'][data['Date'] == '2017-12-25'] = '2017-12-24'
data['Hour'] = [int(i[:2]) for i in data['Time']]
_light = ['S1_Light', 'S2_Light', 'S3_Light', 'S4_Light']
_sound = ['S1_Sound', 'S2_Sound', 'S3_Sound', 'S4_Sound']
_temp = ['S1_Temp', 'S2_Temp', 'S3_Temp', 'S4_Temp']
_features = _light + _sound + _temp
_dates = ['2017-12-22','2017-12-23','2017-12-24','2017-12-26', '2018-01-10',
         '2018-01-11']
_cols = [_light, _sound, _temp]

'''
GRAPHS FOR EACH DAY

for _feature in _features:
    for _date in _dates:
        _a,_b,_c,_d = model_functions.segement_by_date(data, [0,1,2,3], _date)
        sns.scatterplot(x = _a['Hour'], y = _a[_feature], label='0')
        sns.scatterplot(x = _b['Hour'], y = _b[_feature], label='1')
        sns.scatterplot(x = _c['Hour'], y = _c[_feature], label='2')
        sns.scatterplot(x = _d['Hour'], y = _d[_feature], label='3')
        plt.title(_date)
        plt.show()

for _col in _cols:
    model_functions.plot_vals_vs_time(data, _col, _dates)
    
for _col in _cols:
    model_functions.plot_people_vs_vals(data, _col, _dates)
'''
    
for _col in _cols:
    for _item in _col:
        sns.scatterplot(y = data['Room_Occupancy_Count'],
                        x = data[_item], label=_item)
        plt.title('All Days')
        plt.show()
        
'''

No one goes to the room on 24-12-2017, 26-12-2017, 11-01-3018

1. When s6_pir = 1, chance of 0 people = 2.1%
                    chance of 1 person = 17.3%
                    chance of 2 people = 39.2%
                    chance of 3 people = 41.3%
                    
   When s7_pir = 1, chance of 0 people = 1.5%
                    chance of 1 person = 2.6%
                    chance of 2 people = 40.0%
                    chance of 3 people = 56.5%
                    
   When s7_pir = 0, chance of 0 people = 88.1%
                    chance of 1 person = 4.7%
                    chance of 2 people = 4.6%
                    chance of 3 people = 2.6%
                    
   When s6_pir = 0, chance of 0 people = 89.0%
                    chance of 1 person = 3.2%
                    chance of 2 people = 4.2%
                    chance of 3 people = 3.4%

2. When 0 people, all sensors detect low light
   When 1 person, s1 light is high while others remain low
   When 2 people, s1 light remains high and,
                  sometimes s2 becomes high with it or sometimes s3 does
   When 3 people, s1, s2, s3 light are all high. s4 also increases, but it has
   lowest increase 
   On 10-01-2018, even though there are 3 people, all lights are low. They
   also leave soon after the lights turn on. Maybe there was some problem with
   the sensors or lights?

3. b = data['S5_CO2'][data['Room_Occupancy_Count']==0].value_counts()
   When there are no people in the room, the s5_co2 reading stabilizes at
   around 350-360. When people come in, the reading begins to increase 
   noticably, however after they leave, it takes several hours for the 
   reading to go back to the stable reading. 

4. When 0 people, all sensors detect low sound. Sometimes, when 0 people,
   sound > 1.5 is detected by all sensors, but this occurs very few times 
   (max is 6 times for s1).
    
   When 1 person, s2, s3, s4 all remain low. However, the range of s1 
   increases. This along with the s1_light increasing when one person
   enters reinforces that when one person enters, he goes towards
   s1.
    
   When more than two people, all the sensors show increased range for sound.
   Notably, s4 again shows the lowest values. It is probably placed furthest
   from the people.
'''