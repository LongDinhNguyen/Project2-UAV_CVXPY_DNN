
import scipy as sp
# import matplotlib.pyplot as plt
import time
#from sklearn.preprocessing import StandardScaler
#from sklearn.model_selection import train_test_split

import keras as kas
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD, RMSprop, Adam

# sp.random.seed(7)

# importing the dataset of opt algorithm training set
# load dataset and split into input (X) and output (Y) variables
#training_dataset = sp.load('x_OHT_dataset.npz')
#X = training_dataset['chan_dataset']
#y = training_dataset['tau_dataset']

def new_activation(x):
    return sp.min(Activation.relu(x))

def build_dnn_train(X_train, y_train, dimen_input, num_epochs, num_batch_size):

    t0 = time.time()

    # define model
    # note that input_dim only required for first hidden layer
    nn_model = Sequential()   # initialize/reset model
    nn_model.add(Dense(200, input_dim=dimen_input, kernel_initializer='TruncatedNormal', use_bias=True, bias_initializer='zeros', activation='sigmoid'))     # 1st hidden layer (10 neurons, 25 expected input variables)
    nn_model.add(Dropout(0.25))
    nn_model.add(Dense(80, kernel_initializer='TruncatedNormal', use_bias=True, bias_initializer='zeros', activation='sigmoid'))  # 2st hidden layer (8 neurons)
    nn_model.add(Dropout(0.25))
    nn_model.add(Dense(80, kernel_initializer='TruncatedNormal', use_bias=True, bias_initializer='zeros', activation='sigmoid'))  # 3st hidden layer (8 neurons)
    nn_model.add(Dropout(0.25))
    nn_model.add(Dense(1, kernel_initializer='TruncatedNormal', use_bias=True, bias_initializer='zeros', activation='relu'))  # output layer (1 neurons)

    # compile model
    # sgd = SGD(lr=0.001, decay=1e-6, momentum=0.5, nesterov=True)
    rmsprop = RMSprop(lr=0.0005, decay=0.9, epsilon=None, rho=0.9)
    # adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.9, amsgrad=False)

    # nn_model.compile(loss='mean_squared_error', optimizer='sgd')
    # nn_model.compile(loss='mean_squared_error', optimizer='adam')
    # nn_model.compile(loss='binary_crossentropy', optimizer='adagrad')
    nn_model.compile(loss='mean_squared_error', optimizer='rmsprop')

    # fit model
    # nn_model.fit(X_train, y_train, epochs=num_epochs, batch_size=num_batch_size)
    # nn_model.fit(X_train, y_train, epochs=num_epochs, batch_size=num_batch_size,
    #              validation_split=0.2, verbose=0)
    nn_model.fit(X_train, y_train, epochs=num_epochs, batch_size=num_batch_size,
                 validation_split=0.2, validation_data=(X_train, y_train), verbose=0)


    time_sol = (time.time() - t0)
    print "Time for training neural network:", time_sol, "seconds"

    # evaluate model
    loss = nn_model.evaluate(X_train, y_train, batch_size=num_batch_size)

    # return neural model
    return nn_model, loss


