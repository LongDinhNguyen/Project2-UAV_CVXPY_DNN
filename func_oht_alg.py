
from cvxpy import *
import scipy as sp
import ecos
import time

# Maximin-rate optimization algorithm
def oht_alg(d2d_to_d2d_gains_diag, uav_to_d2d_gains, d2d_to_d2d_gains_diff, eta, power_UAV, power_cir_UAV):
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
        prob = Problem(obj_opt, constraints)
        prob.solve(solver=ECOS_BB)

        theta_sol = theta.value
        phi_n_sol = sp.multiply((theta_sol-1)*eta*power_UAV, uav_to_d2d_gains)
        x_rate = sp.matmul(d2d_to_d2d_gains_diag, phi_n_sol)
        term_rate = sp.matmul(sp.transpose(d2d_to_d2d_gains_diff), phi_n_sol) + 1
        rate_sol_ue = sp.divide(sp.log(sp.add(1, sp.divide(x_rate, term_rate))), theta_sol)
        term_pow_iter = sp.subtract(1, sp.divide(1, theta_sol))*eta*power_UAV*sp.add(1, sp.sum(uav_to_d2d_gains)) + power_cir_UAV
        iter_phi.append(t_max.value)
        if iter >= 2:
            epsilon = sp.divide(sp.absolute(sp.subtract(iter_phi[iter - 1], iter_phi[iter - 2])),
                                sp.absolute(iter_phi[iter - 2]))
        iter_EE = sp.divide(sp.multiply(1e3, sp.divide(sp.sum(rate_sol_ue), term_pow_iter)), sp.log(2))

    return iter_EE, theta_sol



