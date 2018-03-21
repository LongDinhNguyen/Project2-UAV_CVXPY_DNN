
import scipy as sp
# import matplotlib.pyplot as plt
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import keras as kas
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
import func_dnn_model

sp.random.seed(7)

# importing the dataset of opt algorithm training set
# load dataset and split into input (X) and output (Y) variables
training_dataset = sp.load('x_OHT_dataset.npz')
X_train = training_dataset['chan_dataset']
y_train = training_dataset['tau_dataset']

num_d2d_pairs = 10
# print y_train

dimen_input = num_d2d_pairs + num_d2d_pairs*num_d2d_pairs
num_epochs = 300
num_batch_size = 1000

# build train neural network
print "Building and training neural network model ..."
nn_model, loss = func_dnn_model.build_dnn_train(X_train, y_train, dimen_input, num_epochs, num_batch_size)
print "loss:", loss

# Saving trained DNN model
nn_model.save('x_DNN_model.h5')
print "Done. Saved"



