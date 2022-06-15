'''Predicting Top Glove Stock Price'''

import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.utils import plot_model
from modules import DataVisualization, DataPreprocessing,ModelCreation,ModelEvaluation
#%% PATHS
LOG_PATH = os.path.join(os.getcwd(),'log')
DATASET_TRAIN_PATH = os.path.join(os.getcwd(),'datasets','Top_Glove_Stock_Price_Train.csv')
DATASET_TEST_PATH = os.path.join(os.getcwd(),'datasets', 'Top_Glove_Stock_Price_Test.csv')
MMS_PATH = os.path.join(os.getcwd(), 'saved_model', 'mms_scaler.pkl')
MODEL_SAVE_PATH = os.path.join(os.getcwd(), 'saved_model', 'model.h5')
#%% EDA
# STEP 1: Data Loading
train_df = pd.read_csv(DATASET_TRAIN_PATH)
test_df = pd.read_csv(DATASET_TEST_PATH, header=None)
train_df = train_df['Open']
test_df = test_df[1]

#%% STEP 2: Data Interpretation
graph = DataVisualization()
graph.plot_initial_trend(train_df)
''' Observation: the DL_trend_graph.png shows there is a dramatic increment of open 
price, and a downward trend at the end of the data.'''

#%% STEP 3: Data Preprocessing
# a) scale data using minmaxscaler
dp = DataPreprocessing()
mms,train_scaled,test_scaled = dp.min_max_scaler(train_df,test_df,MMS_PATH)

#%% STEP 4: Model Creation

window_size = 60
len_train = len(train_df)
len_test = window_size + len(test_df)

# a) split TRAINING set
mc = ModelCreation()
x_train, y_train = mc.split_data(train_scaled,window_size,len_train)

# b) split TESTING set
dataset_full = np.concatenate((train_scaled,test_scaled), axis=0)
test_data = dataset_full[-len_test:]
x_test, y_test = mc.split_data(test_data,window_size,len_test)

# c) build LSTM model
model = mc.lstm_model(x_train)
plot_model(model)

# d) train the model
mc.train_model(LOG_PATH, model, x_train, y_train, epochs=50)

#%% STEP 5: Model Deployment

# a) predict the model on x test
me = ModelEvaluation()
y_true, y_pred = me.model_pred(model, x_test, y_test, mms)

# b) to view the performance of model
graph.plot_performance(y_true,y_pred)

# c) to view the mean percentage absolute error
print('The Mean Absolute Percentage Error',
      (mean_absolute_error(y_true,y_pred)/sum(abs(y_true))) *100,'%')

# d) save the model
model.save(MODEL_SAVE_PATH)
