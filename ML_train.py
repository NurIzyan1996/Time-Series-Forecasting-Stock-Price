'''Predicting Intel Corporation (INTC) Stock Price'''

import pandas as pd
import os 
import pickle
from modules import ModelCreation,ModelEvaluation,DataVisualization
#%% PATH
DATASET_PATH = os.path.join(os.getcwd(), 'datasets', 'INTC.csv')
MODEL_PATH = os.path.join(os.getcwd(), 'saved_model', 'LRmodel.pkl')

#%% STEP 1: Data Loading
df = pd.read_csv(DATASET_PATH)

#%% STEP 2: Data Inpection

# a) check data information
df.info()
'''Observation: there are no null value'''
df.describe()
'''Observation: no significant different between mean value and 50% value'''
df.boxplot()
'''Observation: ML_boxplot.png shows 'Volume' contains outliers'''

# b) plot Stock Prices against Date. 
df.plot(x="Date",y=["Close","Open","High","Low","Adj Close"])
'''Volume is excluded because it has a different scale.
Observation: ML_trend_graph shows that there is a downward trend of stock prices'''

#%% STEP 3: ML Model

# a) split data into train and test
mc = ModelCreation()
train,test,X_train,y_train,X_test,y_test = mc.split_train_test(df.drop(columns=["Date"]),
                                                               window_size=200,
                                                               col_name='Open')

# b) train ML model
model = mc.linear_model(X_train,y_train)

#%% STEP 4: Model Evaluation

# a) prediction of Open price
pred_open = model.predict(X_test)

# b) evaluate the predictions
me = ModelEvaluation()
me.model_performance(model,X_test,y_test,pred_open)
''' the results sow the R-squared is closer to 1 => good prediction,
Seems like the model is doing a very good job at predicting the stock price,
It could be due to that we are using a small dataset with low number of features.'''

# c) update the data
test.insert(6,'Open_prediction',pred_open) 

# d) visualize the model performance
graph = DataVisualization()
graph.plot_performance2(train["Open"],test["Open"],test["Open_prediction"])

# e)save the model for deployment
pickle.dump(model,open(MODEL_PATH,'wb'))