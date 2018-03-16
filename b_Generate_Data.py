
from cvxpy import *
import scipy as sp
import ecos
import matplotlib.pyplot as plt
import time
import func_oht_alg as func_oht

# sp.random.seed(2035)

# ############################################################
# This code is loaded from Full_code_TH
# ############################################################
# Starting main code for EEmax UAV networks
# ############################################################

bandwidth = 1  # MHz
height = 100  # m
eta = 0.5  # EH efficiency
power_UAV = 5000
power_cir_UAV = 4000
atg_a = 11.95
atg_b = 0.136
noise_variance = sp.multiply(sp.multiply(sp.power(10, sp.divide(-130, 10)), bandwidth), 1e6)
d2d_max = 50

max_chan_realizaion = 10

chan_model = sp.load('x_chan_model.npz')
max_uav_to_d2d_gains = chan_model['uav']
max_d2d_to_d2d_gains = chan_model['d2d']

num_d2d_pairs = 5
time_sol_vec_Mon = []
EE_sol_vec_Mon = []
tau_sol_vec_Mon = []

vec_chan_training = []
avg = {}
t0 = time.time()
num_infeasible = 0
# rmin = sp.multiply(0.2, sp.log(2))

#time_sol = []
EE_sol = []
tau_sol = []
for Mon in xrange(max_chan_realizaion):
    try:
        max_d2d_to_d2d_gains_diff = sp.copy(max_d2d_to_d2d_gains[:, :, Mon])
        sp.fill_diagonal(max_d2d_to_d2d_gains_diff, 0)
        max_d2d_to_d2d_gains_diag = sp.subtract(max_d2d_to_d2d_gains[:, :, Mon], max_d2d_to_d2d_gains_diff)

        uav_to_d2d_gains = max_uav_to_d2d_gains[:num_d2d_pairs]
        d2d_to_d2d_gains = max_d2d_to_d2d_gains[:num_d2d_pairs, :num_d2d_pairs, Mon]
        d2d_to_d2d_gains_diff = max_d2d_to_d2d_gains_diff[:num_d2d_pairs, :num_d2d_pairs]
        d2d_to_d2d_gains_diag = sp.subtract(d2d_to_d2d_gains, d2d_to_d2d_gains_diff)

        # vectorize channel training
        trained_chan = d2d_to_d2d_gains.ravel()
        vec_chan_training.append(trained_chan)

        # maximin sum-rate algorithm
        iter_EE, theta_sol = func_oht.oht_alg(d2d_to_d2d_gains_diag, uav_to_d2d_gains, d2d_to_d2d_gains_diff, eta, power_UAV, power_cir_UAV)
        EE_sol.append(iter_EE)
        tau_sol.append(1 - 1/theta_sol)
    except (SolverError, TypeError):
        # pass
        num_infeasible += 1

# Calculate the total time of training datdaset
time_sol = (time.time() - t0)

v1 = sp.array(EE_sol)
EE_sol_vec_Mon.append(sp.mean(v1))
# v2 = sp.array(time_sol)
# time_sol_vec_Mon.append(sp.mean(v2))
v3 = sp.array(tau_sol)
tau_sol_vec_Mon.append(sp.mean(v3))

vec_tau_training = tau_sol

#print EE_sol_vec_Mon
#print tau_sol_vec_Mon
print "Time for generate training dataset:", time_sol, "seconds"
print "Number of infeasible solving", num_infeasible
print "Size of vec_chan_training", sp.shape(vec_chan_training)
print "Size of vec_tau_training", sp.shape(vec_tau_training)

test_vec_chan_training = vec_chan_training[1][1]

# Saving training dataset for DNN model
sp.savez('x_OHT_dataset', chan_dataset=vec_chan_training, tau_dataset=vec_tau_training)

