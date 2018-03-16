
from cvxpy import *
import scipy as sp
import ecos
import matplotlib.pyplot as plt
import time

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
max_num_d2d_pairs = 10

chan_model = sp.load('chan_model.npz')
max_uav_to_d2d_gains = chan_model['uav']
max_d2d_to_d2d_gains = chan_model['d2d']


# ############################################################
# This loop for a range of num_d2d_pairs
# ############################################################
# range_num_d2d_pairs = [8]
range_num_d2d_pairs = [2, 3, 4, 5, 6, 7, 8, 9, 10]
time_sol_vec_Mon = []
EE_sol_vec_Mon = []
tau_sol_vec_Mon = []

avg = {}
num_infeasible = sp.zeros(len(range_num_d2d_pairs))
for prin in range_num_d2d_pairs:
    num_d2d_pairs = prin
    # rmin = sp.multiply(0.2, sp.log(2))
    time_sol_vec = []
    EE_sol_vec = []
    tau_sol_vec = []

    for Mon in xrange(max_chan_realizaion):
        try:
            max_d2d_to_d2d_gains_diff = sp.copy(max_d2d_to_d2d_gains[:, :, Mon])
            sp.fill_diagonal(max_d2d_to_d2d_gains_diff, 0)
            max_d2d_to_d2d_gains_diag = sp.subtract(max_d2d_to_d2d_gains[:, :, Mon], max_d2d_to_d2d_gains_diff)

            uav_to_d2d_gains = max_uav_to_d2d_gains[:num_d2d_pairs, Mon]
            d2d_to_d2d_gains = max_d2d_to_d2d_gains[:num_d2d_pairs, :num_d2d_pairs, Mon]
            d2d_to_d2d_gains_diff = max_d2d_to_d2d_gains_diff[:num_d2d_pairs, :num_d2d_pairs]
            d2d_to_d2d_gains_diag = sp.subtract(d2d_to_d2d_gains, d2d_to_d2d_gains_diff)

            # ############################################################
            # This code is used to solve the maximin sum-rate problem
            # ############################################################
            t0 = time.time()
            theta_ini = Parameter(value=1/0.5)

            iter = 0
            epsilon = 1
            theta_sol = 0
            iter_phi = []

            while epsilon >= 1e-2 and iter <= 20:
                iter += 1
                if iter == 1:
                   theta_ref = theta_ini.value
                else:
                   theta_ref = theta_sol

                term_x = sp.divide(1, sp.multiply(sp.subtract(theta_ref, 1), sp.matmul(d2d_to_d2d_gains_diag, uav_to_d2d_gains)))
                term_y = sp.add(sp.multiply(sp.subtract(theta_ref, 1), sp.matmul(sp.transpose(d2d_to_d2d_gains_diff), uav_to_d2d_gains)), sp.divide(1, eta * power_UAV))

                a_1 = sp.add(sp.divide(sp.multiply(2, sp.log(sp.add(1, sp.divide(1, sp.multiply(term_x, term_y))))), theta_ref),
                             sp.divide(2, sp.multiply(theta_ref, sp.add(sp.multiply(term_x, term_y), 1))))
                b_1 = sp.divide(1, sp.multiply(theta_ref, sp.multiply(term_x, sp.add(sp.multiply(term_x, term_y), 1))))
                c_1 = sp.divide(1, sp.multiply(theta_ref, sp.multiply(term_y, sp.add(sp.multiply(term_x, term_y), 1))))
                d_1 = sp.divide(sp.log(sp.add(1, sp.divide(1, sp.multiply(term_x, term_y)))), sp.square(theta_ref))

                theta = NonNegative(1)
                t_max = NonNegative(1)

                obj_opt = Maximize(t_max)

                constraints = [theta >= 1]
                constraints.append(
                    t_max <= a_1 - sp.divide(b_1, sp.matmul(d2d_to_d2d_gains_diag, uav_to_d2d_gains)) * inv_pos(theta - 1)
                    - mul_elemwise(c_1, sp.matmul(sp.transpose(d2d_to_d2d_gains_diff), uav_to_d2d_gains) * (theta - 1) + sp.divide(1, eta * power_UAV))
                    - d_1 * theta)

                t1 = time.time()

                prob = Problem(obj_opt, constraints)
                prob.solve(solver=ECOS_BB)

                # print 'Iteration:', iter, '; Time:', time.time() - t1

                theta_sol = theta.value
                phi_n_sol = sp.multiply((theta_sol-1)*eta*power_UAV, uav_to_d2d_gains)

                x_rate = sp.matmul(d2d_to_d2d_gains_diag, phi_n_sol)
                term_rate = sp.matmul(sp.transpose(d2d_to_d2d_gains_diff), phi_n_sol) + 1
                rate_sol_ue = sp.divide(sp.log(sp.add(1, sp.divide(x_rate, term_rate))), theta_sol)

                # print rate_sol_ue

                term_pow_iter = sp.subtract(1, sp.divide(1, theta_sol))*eta*power_UAV*sp.add(1, sp.sum(uav_to_d2d_gains)) + power_cir_UAV


                iter_phi.append(t_max.value)

                if iter >= 2:
                    epsilon = sp.divide(sp.absolute(sp.subtract(iter_phi[iter - 1], iter_phi[iter - 2])),
                                        sp.absolute(iter_phi[iter - 2]))

                iter_EE = sp.divide(sp.multiply(1e3, sp.divide(sp.sum(rate_sol_ue), term_pow_iter)), sp.log(2))

            # print 'Number of iterations:', iter
            # print 'Solution:', vars.value
            EE_sol_vec.append(iter_EE)
            tau_sol_vec.append(1 - 1/theta_sol)
            time_sol = (time.time() - t0)
            time_sol_vec.append(time_sol)

        except (SolverError, TypeError):
            # pass
            num_infeasible[prin - 2] += 1

    v1 = sp.array(EE_sol_vec)
    EE_sol_vec_Mon.append(sp.mean(v1))
    v2 = sp.array(time_sol_vec)
    time_sol_vec_Mon.append(sp.mean(v2))

    v3 = sp.array(tau_sol_vec)
    tau_sol_vec_Mon.append(sp.mean(v3))

print EE_sol_vec_Mon
print time_sol_vec_Mon
print tau_sol_vec_Mon

print num_infeasible

sp.savez('result_HT', EE_TH=EE_sol_vec_Mon, time_TH=time_sol_vec_Mon, tau_TH=tau_sol_vec_Mon)

# plt.figure(figsize=(8, 6))
# plt.clf()
# plt.plot(range_num_d2d_pairs, time_sol_vec_Mon)
# plt.figure(figsize=(8, 6))
# plt.clf()
# plt.plot(range_num_d2d_pairs, EE_sol_vec_Mon)
# plt.show()
