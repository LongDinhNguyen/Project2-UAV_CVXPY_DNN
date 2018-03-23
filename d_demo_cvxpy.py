
import os
from cvxpy import *
import scipy as sp
import ecos
#import matplotlib.pyplot as plt
import time
import func_oht_alg as func_oht

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

num_d2d_pairs = 2

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

