# Room_Occupancy_Estimation
Using various approaches to predict the number of people in a room based on values recorded by different sensors.
The dataset was downloaded from this link: https://archive.ics.uci.edu/ml/datasets/Room+Occupancy+Estimation#

A brief description of the dataset:

The dataset contains columns for 4 temperature sensors, 4 sound sensors, 4 light sensors, 2 motion (PIR) sensors,
2 carbon dioxide sensors, and 1 column for the target, Room_Occupancy_Count, which ranges from 0 to 3. The
readings of the sensors are used to make predictions for the target.

Each file contains a different approach to solving the problem, except model_functions.py, which contains
utility functions (some of which are unrelated to this project), and Occupancy_Estimation.csv, which is the dataset used. 


A brief description of the files follows:

1. occupancy_data_ar.py : Uses an Autoregressive Model
2. occupancy_data_linreg.py : Uses Linear Regression
3. occupancy_data_logreg.py : Uses Logistic Regression
4. occupancy_data_exploration.py : Contains a simple framework for EDA
5. occupancy_data_exploration_2.py : Contains a better framework for EDA
6. club_2_and_3.py : Here, we treat a room occupancy count of 2 the same as a room occupancy count of 3. I've
   used Linear and Logistic Regression, Decision Trees and Random Forest, as well as Adaptive Boosting.

Ironically, it is the Autoregressive Model (arguably the simplest one), which performs best. 
