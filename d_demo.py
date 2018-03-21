
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from cvxpy import *
import scipy as sp
import ecos
#import matplotlib.pyplot as plt
import time
import func_oht_alg as func_oht


import keras as kas
from keras.models import load_model
# import func_dnn_model

sp.random.seed(7)

bandwidth = 1  # MHz
height = 100  # m
eta = 0.5  # EH efficiency
power_UAV = 5000
power_cir_UAV = 4000
atg_a = 11.95
atg_b = 0.136
noise_variance = sp.multiply(sp.multiply(sp.power(10, sp.divide(-130, 10)), bandwidth), 1e6)
d2d_max = 50

# importing the testing channel into input (X) and predict output (y) (optimal tau)
chan_model = sp.load('x_chan_model_test.npz')
max_uav_to_d2d_gains = chan_model['uav']
max_d2d_to_d2d_gains = chan_model['d2d']

num_d2d_pairs = 10

max_chan_realizaion = 100

# #########################################################
# This part is use traditional OPT alg for OHT sovling problem
vec_chan = []
avg = {}


t0 = time.time()
num_infeasible = 0
# rmin = sp.multiply(0.2, sp.log(2))

print "solving problem with OHT-OPT Alg ..."

EE_sol_vec_Mon = []
maximin_rate_sol_Mon = []
tau_sol_vec_Mon = []

maximin_rate = []
EE_sol = []
tau_sol = []
for Mon in xrange(max_chan_realizaion):
    try:
        max_d2d_to_d2d_gains_diff = sp.copy(max_d2d_to_d2d_gains[:, :, Mon])
        sp.fill_diagonal(max_d2d_to_d2d_gains_diff, 0)
        max_d2d_to_d2d_gains_diag = sp.subtract(max_d2d_to_d2d_gains[:, :, Mon], max_d2d_to_d2d_gains_diff)

        uav_to_d2d_gains = max_uav_to_d2d_gains[:num_d2d_pairs, Mon]
        d2d_to_d2d_gains = max_d2d_to_d2d_gains[:num_d2d_pairs, :num_d2d_pairs, Mon]
        d2d_to_d2d_gains_diff = max_d2d_to_d2d_gains_diff[:num_d2d_pairs, :num_d2d_pairs]
        d2d_to_d2d_gains_diag = sp.subtract(d2d_to_d2d_gains, d2d_to_d2d_gains_diff)

        # maximin sum-rate algorithm
        iter_EE, theta_sol, iter_maximin_rate = func_oht.oht_alg(d2d_to_d2d_gains_diag, uav_to_d2d_gains, d2d_to_d2d_gains_diff, eta, power_UAV, power_cir_UAV)
        EE_sol.append(iter_EE)
        tau_sol.append(1 - 1/theta_sol)
        maximin_rate.append(iter_maximin_rate)

    except (SolverError, TypeError):
        # pass
        num_infeasible += 1

# Calculate the total time of solving
time_sol = (time.time() - t0)

v1 = sp.array(EE_sol)
EE_sol_vec_Mon = sp.mean(v1)
v2 = sp.array(maximin_rate)
maximin_rate_sol_Mon.append(sp.mean(v2))
v3 = sp.array(tau_sol)
tau_sol_vec_Mon = sp.mean(v3)

print "EE from OHT-OPT:", EE_sol_vec_Mon
print "tau_value from OHT-OPT", tau_sol_vec_Mon
print "maximin rate from OHT-OPT", maximin_rate_sol_Mon
print "Solving time for OHT-OPT Alg:", time_sol, "seconds"


print "# ################################################ #"


# #########################################################
# This part is use DNN model for sovling OHT problem

# import trained DNN
nn_model = load_model('x_DNN_model.h5')
print "Imported DNN model"

# testing with a realization channel
# X_test = sp.rand(1, 30)
# test_result = nn_model.predict(sp.array(X_test), verbose=1)
# test_result = nn_model.predict_classes(X_test, verbose=1)

dimen_input = num_d2d_pairs + num_d2d_pairs*num_d2d_pairs

t0 = time.time()

print "solving problem with DNN model ..."

Maxmin_rate_test = []
EE_sol_test = []
tau_sol_test = []
for Mon in xrange(max_chan_realizaion):
    max_d2d_to_d2d_gains_diff = sp.copy(max_d2d_to_d2d_gains[:, :, Mon])
    sp.fill_diagonal(max_d2d_to_d2d_gains_diff, 0)

    uav_to_d2d_gains = max_uav_to_d2d_gains[:num_d2d_pairs, Mon]
    d2d_to_d2d_gains = max_d2d_to_d2d_gains[:num_d2d_pairs, :num_d2d_pairs, Mon]
    d2d_to_d2d_gains_diff = max_d2d_to_d2d_gains_diff[:num_d2d_pairs, :num_d2d_pairs]
    d2d_to_d2d_gains_diag = sp.subtract(d2d_to_d2d_gains, d2d_to_d2d_gains_diff)

    # vectorize channel training
    test_chan = sp.zeros(dimen_input)
    test_chan[0:num_d2d_pairs] = uav_to_d2d_gains
    test_chan[num_d2d_pairs:dimen_input] = d2d_to_d2d_gains.ravel()

    vec_chan_test = sp.array([test_chan])
    X_test = vec_chan_test

    test_tau_result = nn_model.predict(X_test, verbose=0)

    test_theta_dnn = 1/(1-test_tau_result)

    phi_n_sol = sp.multiply((test_theta_dnn - 1) * eta * power_UAV, uav_to_d2d_gains)
    x_rate = sp.matmul(d2d_to_d2d_gains_diag, sp.transpose(phi_n_sol))
    term_rate = sp.matmul(sp.transpose(d2d_to_d2d_gains_diff), sp.transpose(phi_n_sol)) + 1
    rate_sol_ue = sp.divide(sp.log(sp.add(1, sp.divide(x_rate, term_rate))), test_theta_dnn)
    maximin_rate_test = min(rate_sol_ue)
    term_pow_iter = sp.subtract(1, sp.divide(1, test_theta_dnn)) * eta * power_UAV * sp.add(1, sp.sum(uav_to_d2d_gains)) + power_cir_UAV

    iter_EE_test = sp.divide(sp.multiply(1e3, sp.divide(sp.sum(rate_sol_ue), term_pow_iter)), sp.log(2))
    EE_sol_test.append(iter_EE_test)
    tau_sol_test.append(test_tau_result)
    Maxmin_rate_test.append(maximin_rate_test)

# Calculate the total time of solving
time_sol = (time.time() - t0)


v1_test = sp.array(EE_sol_test)
EE_sol_DNN_test = sp.mean(v1_test)
v2_test = sp.array(tau_sol_test)
tau_sol_DNN_test = sp.mean(v2_test)
v3_test = sp.array(Maxmin_rate_test)
Maximin_sol_DNN_test = sp.mean(v3_test)

print "EE from DNN:", EE_sol_DNN_test
print "tau_value from DNN:", tau_sol_DNN_test
print "maximin rate from DNN:", Maximin_sol_DNN_test
print "Solving time for OHT-OPT Alg:", time_sol, "seconds"

