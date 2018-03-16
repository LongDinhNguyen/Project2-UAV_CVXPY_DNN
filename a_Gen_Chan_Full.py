from cvxpy import *
import scipy as sp
import ecos
import matplotlib.pyplot as plt
import time

# sp.random.seed(2030)

# ############################################################
# This code is used to set up the UAV model
# ############################################################

coverage_r = 1000 #m
plt.figure(figsize=(10, 10))
coverage = plt.Circle((0, 0), coverage_r, color='green', fill=False)
ax = plt.gca()
ax.add_patch(coverage)

class UAV(object):

    def __init__(self, h):
        self.h = h
        plt.scatter(0, 0, s=200, c='black', marker='D')

    def loss_to_pair(self, pair, atg_a, atg_b, pl_exp=4, gamma=1e2):
        dist = sp.sqrt(sp.add(sp.square(pair.tx_x), sp.add(sp.square(pair.tx_y), sp.square(self.h))))
        phi = sp.multiply(sp.divide(180, sp.pi), sp.arcsin(sp.divide(self.h, dist)))
        pr_LOS = sp.divide(1, sp.add(1, sp.multiply(atg_a, sp.exp(sp.multiply(-atg_b, sp.subtract(phi, atg_a))))))
        pr_NLOS = sp.subtract(1, pr_LOS)

        total_loss = sp.add(sp.multiply(pr_LOS, sp.power(dist, -pl_exp)),
                            sp.multiply(sp.multiply(pr_NLOS, gamma), sp.power(dist, -pl_exp)))

        return total_loss

class D2DPair(object):

    def __init__(self, id, coverage_r, max_dist, origin_x=0, origin_y=0, low_tx=0.2, low_rx=0.5):
    # def __init__(self, id, coverage_r, max_dist, origin_x=0, origin_y=0, low_tx=0.3, low_rx=0.2):
        tx_d = sp.multiply(coverage_r, sp.random.uniform(low=low_tx))
        tx_angle = sp.multiply(2, sp.multiply(sp.pi, sp.subtract(sp.random.rand(), 1)))
        self.tx_x = sp.add(origin_x, sp.multiply(tx_d, sp.sin(tx_angle)))
        self.tx_y = sp.add(origin_y, sp.multiply(tx_d, sp.cos(tx_angle)))

        d2d_d = sp.multiply(max_dist, sp.random.uniform(low=low_rx))
        d2d_angle = sp.multiply(2, sp.multiply(sp.pi, sp.subtract(sp.random.rand(), 1)))
        self.rx_x = sp.add(self.tx_x, sp.multiply(d2d_d, sp.sin(d2d_angle)))
        self.rx_y = sp.add(self.tx_y, sp.multiply(d2d_d, sp.cos(d2d_angle)))

        plt.scatter(self.tx_x, self.tx_y, s=20, c='red')
        plt.scatter(self.rx_x, self.rx_y, s=20, c='blue')
        plt.annotate(id, (self.tx_x + 10, self.tx_y + 10))

    def loss_to_pair(self, pair, gain=1e-3, exp_factor=sp.random.exponential(1), pl_exp=3):
        dist = sp.sqrt(sp.add(sp.square(sp.subtract(self.tx_x, pair.rx_x)),
                              sp.square(sp.subtract(self.tx_y, pair.rx_y))))
        loss = sp.multiply(gain, sp.multiply(sp.square(exp_factor), sp.power(dist, -pl_exp)))

        return loss


# ############################################################
# Starting main code for EEmax UAV networks
# ############################################################

bandwidth = 1 #MHz
height = 200 #m
# eta = 0.5 #EH efficiency
# power_UAV = 5000
# power_cir_UAV = 4000
atg_a = 11.95
atg_b = 0.136
noise_variance = sp.multiply(sp.multiply(sp.power(10, sp.divide(-130, 10)), bandwidth), 1e6)
# noise_variance = sp.power(10, sp.divide(-109, 10))
d2d_max = 50

max_chan_realizaion = 100
max_num_d2d_pairs = 10

max_uav_to_d2d_gains = sp.zeros((max_num_d2d_pairs, max_chan_realizaion))
max_d2d_to_d2d_gains = sp.zeros((max_num_d2d_pairs, max_num_d2d_pairs, max_chan_realizaion))
# ############################################################
# This loop for channel realization - Monte Carlos
# ############################################################
for Mon in xrange(max_chan_realizaion):
    d2d_pairs = []
    uav = UAV(height)
    for p in range(max_num_d2d_pairs):
        # d2d_pairs.append(D2DPair(p, coverage_r, d2d_max, low_rx=0.8))
        # d2d_pairs.append(D2DPair(p, coverage_r, d2d_max))
        d2d_pairs.append(D2DPair(p, coverage_r, d2d_max, low_rx=0.0, low_tx=0.0))

    for i in xrange(max_num_d2d_pairs):
        for j in xrange(max_num_d2d_pairs):
            max_d2d_to_d2d_gains[i, j, Mon] = sp.divide(d2d_pairs[i].loss_to_pair(d2d_pairs[j]), noise_variance)
        max_uav_to_d2d_gains[i, Mon] = sp.divide(uav.loss_to_pair(d2d_pairs[i], atg_a, atg_b), noise_variance)

# print max_uav_to_d2d_gains
# print max_d2d_to_d2d_gains
    # max_d2d_to_d2d_gains_diff = sp.copy(max_d2d_to_d2d_gains)
    # sp.fill_diagonal(max_d2d_to_d2d_gains_diff, 0)
    # max_d2d_to_d2d_gains_diag = sp.subtract(max_d2d_to_d2d_gains, max_d2d_to_d2d_gains_diff)

sp.savez('chan_model', uav=max_uav_to_d2d_gains, d2d=max_d2d_to_d2d_gains)

# plt.show()