#Importing initial libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing Samsung Stock Price as Dataset
dataset = pd.read_csv('Samsung_SP_Train.csv')
past_prices = dataset.iloc[:,1:2].values

#Scaling training data 
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
X_scaled = sc.fit_transform(past_prices)

#Creating training matrix with a timestep of 90 days
X_train = []
y_train = []
for i in range(90,len(dataset)):
    X_train.append(X_scaled[i-90:i,0])
    y_train.append(X_scaled[i,0])
X_train, y_train = np.array(X_train), np.array(y_train)


   
#Import LG Stock Price as additional feature

# dataset_lg = pd.read_csv('LG_SP_Train.csv')
# X_lg = dataset_lg.iloc[90:,1:2].values

#Combine Stock Prices to 3D Tensor for RNN Input
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

#Build RNN
#Import libraries
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

#Creating first layer of RNN
regressor = Sequential()
regressor.add(LSTM(units = 50, return_sequences = 'True', input_shape = (X_train.shape[1], 1) ))
regressor.add(Dropout(0.2))

#Second layer of RNN
regressor.add(LSTM(units = 50, return_sequences = 'True' ))
regressor.add(Dropout(0.2))

#Final layer of RNN
regressor.add(LSTM(units = 50 ))
regressor.add(Dropout(0.2))

#Last layer of NN
regressor.add(Dense(units = 1))

#Compile RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

#Fit RNN with training set
regressor.fit(X_train, y_train, batch_size = 32, epochs = 10)

#Evaluate RNN
"""
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score

def build_regressor():
    regressor = Sequential()
    regressor.add(LSTM(units = 50, return_sequences = 'True', input_shape = (X_train.shape[1], 1) ))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units = 50, return_sequences = 'True' ))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units = 50 ))
    regressor.add(Dropout(0.2))
    regressor.add(Dense(units = 1))
    regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
    return regressor
regressor = KerasRegressor(build_fn = build_regressor, batch_size = 32, epochs = 40)
accuracies = cross_val_score(estimator = regressor, X = X_train, y = y_train, cv = 10)
average = accuracies.mean()
variance = accuracies.std()
"""

#Refine RNN - The following are methods to take when refining our RNN
"""
from sklearn.model_selection import GridSearchCV

#Build regressor to be used in GridSearch 
def build_regressor():
    regressor = Sequential()
    regressor.add(LSTM(units = 50, return_sequences = 'True', input_shape = (X_train.shape[1], 1) ))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units = 50, return_sequences = 'True' ))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units = 50 ))
    regressor.add(Dropout(0.2))
    regressor.add(Dense(units = 1))
    regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
    return regressor
regressor = KerasRegressor(build_fn = build_regressor)

#Choose parameters to be used in GridSearch
parameters = {'batch_size' : [32,64],
              'epochs' : [100,150]}

#Create GridSearch
grid = GridSearchCV(estimator = regressor, 
                    param_grid = parameters,
                    scoring = 'neg_mean_squared_error',
                    cv = 4)

#Run GridSearch to output optimal parameters
grid_search = grid.fit(X_train, y_train)
best_accuracy = grid.best_score_
best_parameters = grid.best_params_
"""

#Import Unseen Samsung Stock Prices
testset = pd.read_csv('Samsung_SP_Test.csv')
future_prices = testset.iloc[:,1:2].values

#Combine Past and Unseen Stock Prices to produce total dataset to allow prediction
totalset = np.concatenate((past_prices, future_prices), axis = 0)
inputs = totalset[len(past_prices)-90:,0]
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)

#Create Test matrix
X_test = []
for i in range(90, len(inputs)):
    X_test.append(inputs[i-90:i,0])
X_test = np.array(X_test)  
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))  

#Predict future prices
y_pred = regressor.predict(X_test)
predicted_prices = sc.inverse_transform(y_pred)

#Visualise predictions against actual prices
plt.plot(predicted_prices, color = 'green', label = 'Predicted Stock Price')
plt.plot(future_prices, color = 'red', label = 'Actual Stock Price')
plt.title('Samsung Stock Prices Predictions')
plt.xlabel('Time')
plt.ylabel('Stock Prices')
plt.legend()
plt.show()
