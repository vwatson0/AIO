import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import  Matern
import scipy
from scipy import stats
import matplotlib.pyplot as plt

import OptimLibV0 as Opt

'''Run Kalman filter on dumb example using the OptimLibV0 function'''

N = 1000
time = np.linspace(0, 100, N)
pos = np.arctan(time/10 - 5)*30
sig = 2
mes = pos + np.random.randn(len(pos)) * sig # creating the measure vector

Pos = np.random.randint(300, size = 10) + 700 # adding peaks to challenge the system
mes[Pos] += 20


#initializing KF
KF = Opt.KFpar(10E-2, 10E+2) # Confidence in lin dynamic, confidence in measurement (Adjust the sensitivity of the filter)

KF.X[0] = mes[0]

esty = np.zeros(len(time))# estimated beam current
estdy = np.zeros(len(time))# estimated first order
ests = np.zeros(len(time))# estimated std

for k in range(N-1):
    deltaT = .1
    KF.F[0, 1] = deltaT

    KF = Opt.PredictCurrentValue(KF, mes[k+1])
    KF.X = KF.X[0] # for unknown reason KF.X is returned encapsulated in another dimension
    esty[k+1] = KF.X[0]
    estdy[k+1] = KF.X[1]
    ests[k+1] = KF.Sig[0]

thresh = .8

plt.figure()
plt.subplot(311)
plt.plot(time, mes)
plt.plot(time, pos)
plt.plot(time, esty)
plt.legend(['measured', 'real', 'filtered'])
plt.ylabel('x')
plt.subplot(312)
plt.plot(time, ests)
plt.ylabel(r'$\hat{\sigma}_x$')
plt.subplot(313)
plt.plot(time, estdy)
plt.plot(time, thresh * np.ones(len(time)), '--')
plt.ylabel(r'$\frac{dx}{dt}$')
plt.xlabel('time')
plt.show()