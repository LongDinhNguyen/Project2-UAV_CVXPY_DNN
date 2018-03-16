
import scipy as sp
import matplotlib.pyplot as plt
import time
#from sklearn.preprocessing import StandardScaler
#from sklearn.model_selection import train_test_split

import keras as kas
from keras.models import Sequential
from keras.layers import Dense

sp.random.seed(7)

# importing the dataset of opt algorithm training set
# load dataset and split into input (X) and output (Y) variables
training_dataset = sp.load('x_OHT_dataset.npz')
X = training_dataset['chan_dataset']
y = training_dataset['tau_dataset']

print y

# define model
# note that input_dim only required for first hidden layer
nn_model = Sequential()   # initialize/reset model
nn_model.add(Dense(25, input_dim=8, init='uniform', activation='relu'))     # 1st hidden layer (10 neurons, 25 expected input variables)
nn_model.add(Dense(10, init='uniform', activation='relu'))  # 2st hidden layer (8 neurons)
nn_model.add(Dense(1, init='uniform', activation='sigmoid'))  # output layer (1 neurons)

# compile model
nn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit model
nn_model.fit(X, y, epochs=100, batch_size=10)

# evaluate model
eval_model = nn_model.evaluate(X, y)


