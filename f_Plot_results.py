import scipy as sp
import matplotlib.pyplot as plt


Res_PA05 = sp.load('result_PA_tau05.npz')
EE_sol_PA05 = Res_PA05['EE_PA']
time_sol_PA05 = Res_PA05['time_PA']

Plot_x_axis = Res_PA05['x_axis']

# Res_PA07 = sp.load('result_PA_tau07.npz')
# EE_sol_PA07 = Res_PA07['EE_PA']
# time_sol_PA07 = Res_PA07['time_PA']
#
# Res_PA03 = sp.load('result_PA_tau03.npz')
# EE_sol_PA03 = Res_PA03['EE_PA']
# time_sol_PA03 = Res_PA03['time_PA']

Res_HT = sp.load('result_HT.npz')
EE_sol_HT = Res_HT['EE_TH']
time_sol_HT = Res_HT['time_TH']

Res_JHTPA = sp.load('result_JHTPA.npz')
EE_sol_JHTPA = Res_JHTPA['EE_JTHPA']
time_sol_JHTPA = Res_JHTPA['time_JTHPA']



# Plot running_time
plt.figure(figsize=(8, 6))
plt.clf()
plt.plot(Plot_x_axis, 1e3*time_sol_PA05, 'r-*', label=r'OPA$(\tau=0.5$)')
plt.plot(Plot_x_axis, 1e3*time_sol_HT, 'b-s', label='OHT')
plt.plot(Plot_x_axis, 1e3*time_sol_JHTPA, 'g-^', label='JHTPA')
# plt.legend(handles=[line1, line2, line3])
plt.legend(loc='upper left', fontsize=10)
plt.ylim(0, max(1e3*time_sol_JHTPA)+10)
plt.ylabel('Running time (milliseconds)', fontsize=14)
plt.xlabel('Number of D2D pairs', fontsize=14)
plt.grid(True)


# Plot EE performance
plt.figure(figsize=(8, 6))
plt.clf()
plt.plot(Plot_x_axis, EE_sol_PA05, 'r-*', label=r'OPA$(\tau=0.5$)')
# plt.plot(Plot_x_axis, EE_sol_PA07, 'r--*', label=r'OPA$(\tau=0.3$)')
# plt.plot(Plot_x_axis, EE_sol_PA03, 'r:d', label=r'OPA$(\tau=0.7$)')
plt.plot(Plot_x_axis, EE_sol_HT, 'b-s', label='OHT')
plt.plot(Plot_x_axis, EE_sol_JHTPA, 'g-^', label='JHTPA')
plt.legend(loc='upper left', fontsize=10)
plt.ylim(min(EE_sol_HT)-0.1, max(EE_sol_JHTPA)+0.1)
# plt.ylim(0, 0.9)
plt.ylabel('EE performance (bits/J/Hz)', fontsize=14)
plt.xlabel('Number of D2D pairs', fontsize=14)
plt.grid(True)

plt.show()

