import os
import datetime
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense,Dropout,LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import TensorBoard

class DataVisualization():
    
    def __init__(self):
        pass
    
    def plot_initial_trend(self, data):
        plt.figure()
        plt.plot(data)
        return plt.show()
    
    def plot_performance(self, data_1, data_2):
        
        plt.figure()
        plt.plot(data_1, color='r', label='Actual Stock Price')
        plt.plot(data_2, color='b', label='Predicted Stock Price')
        plt.legend(['Actual','Predicted'])
        return plt.show()
    
class DataPreprocessing():
    
    def __init__(self):
        pass
    
    def min_max_scaler(self,data_1,data_2,path):
        mms = MinMaxScaler()
        new_data_1 = mms.fit_transform(np.expand_dims(data_1,-1))
        new_data_2 = mms.transform(np.expand_dims(data_1,-1))
        pickle.dump(mms, open(path, 'wb'))
        return mms,new_data_1,new_data_2
    
class ModelCreation():
    def __init__(self):
        pass
    
    def split_data(self, data, size_1, size_2):
        data_x = np.array([data[i-size_1:i,0] for i in range(size_1,size_2)])
        data_x = np.expand_dims(data_x,-1)
        data_y = np.array([data[i,0] for i in range(size_1,size_2)])
        return data_x, data_y
    
    def lstm_model(self,data):
        model = Sequential()
        model.add(LSTM(128, activation='tanh',
                       return_sequences=True, 
                       input_shape=(data.shape[1:]))) 
        model.add(Dropout(0.2))
        model.add(LSTM(128, activation='tanh'))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.summary()
        model.compile(optimizer='adam',loss='mse',metrics='mse')
        return model
    
    def train_model(self, path, model, data_1, data_2, epochs):
        log_files = os.path.join(path,datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
        tensorboard_callback = TensorBoard(log_dir=log_files, histogram_freq=1)
        return model.fit(data_1, data_2, epochs=epochs, 
                         callbacks=tensorboard_callback)
    
class ModelEvaluation():
    def __init__(self):
        pass

    def model_pred(self, model, data_1, data_2, scaler):
        predicted = np.array([model.predict(np.expand_dims(test,axis=0)) 
                          for test in data_1])
        new_data_1 = scaler.inverse_transform(np.expand_dims(data_2,axis=-1))
        new_data_2 = scaler.inverse_transform(predicted.reshape(len(predicted),
                                                                1))
        return new_data_1, new_data_2
