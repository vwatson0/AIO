import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import scipy
from scipy import stats
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

import OptimLibV0 as Opt

''' This script is testing the optimizer function GetNextOptimum form library OptimLibV0
requirement: OptimLibV0.py'''

def model(coord, tet, amp=1):
    ''' Sumulating a dummy VENUS '''
    metasection = 0

    for k in range(len(tet)):
        section = 0
        for l in range(len(coord)):
            section += -((coord[l] - tet[k, l]) / (tet[k, 2*l+1])) ** 2
        metasection += np.exp(section)

    return amp * metasection


Dim = 2
time = 1000

# creating data
#np.random.seed(0)

# parameters of the function at the begining [amaxX, amaxt, stdX, stdY, A]
ThetaCritFuncPos = np.zeros([5, int(2*Dim+2)])  # Parametwers of <Ib>
ThetaCritFuncPos[:, 0:int(Dim)] = np.random.rand(5, int(Dim)) * 8 + 1  # x peak loc
ThetaCritFuncPos[:, int(Dim)] = np.random.rand(5) * 800 + 100  # t peak loc
#ThetaCritFuncPos[0, int(Dim)] = 0.
ThetaCritFuncPos[:, int(Dim+1):int(2*Dim+1)] = np.random.rand(5, int(Dim)) * 3 + 1  # x peak sig
ThetaCritFuncPos[:, int(2*Dim+1)] = np.random.rand(5) * 80 + 100  # t peak sig
ThetaCritFuncNeg = np.zeros([3,  int(2*Dim+2)])  # Parametwers of <Ib>
ThetaCritFuncNeg[:, 0:int(Dim)] = np.random.rand(3, int(Dim)) * 10  # x peak loc
ThetaCritFuncNeg[:, int(Dim)] = np.random.rand(3) * 1000  # t peak loc
ThetaCritFuncNeg[:, int(Dim+1):int(2*Dim+1)] = np.random.rand(3, int(Dim)) * 1 + 1  # x peak sig
ThetaCritFuncNeg[:, int(2*Dim+1)] = np.random.rand(3) * 80 + 100  # t peak sig
ThetaControl = np.zeros([2, int(2*Dim+2)])  # Parameters of \sig_{Ib}
#ThetaControl[:, 0:int(Dim)] = np.random.rand(3, int(Dim)) * 10  # x peak loc
ThetaControl[:, 0:int(Dim)] = ThetaCritFuncPos[1:3, 0:int(Dim)] # x peak loc
#ThetaControl[:, int(Dim)] = np.random.rand(3) * 800 + 100  # t peak loc
ThetaControl[:, int(Dim)] = ThetaCritFuncPos[1:3, int(Dim)]  # t peak loc
ThetaControl[:, int(Dim+1):int(2*Dim+1)] = np.random.rand(2, int(Dim)) * 1 + 1
ThetaControl[:, int(2*Dim+1)] = np.random.rand(2) * 100 + 30


# 15 random point then start optim

coord = np.zeros([Dim+1, time])
timeVect = np.zeros(time)
evalFunc = np.zeros(time)
evalControl = np.zeros(time)
risk = np.zeros([1000, 2])


# get best worse most stable most unstable paths to see how it goes.
MaxevalFunc = np.zeros(time)
MinevalFunc = np.zeros(time)
MaxevalControl = np.zeros(time)
MinevalControl = np.zeros(time)

Amax = np.zeros([time, Dim])

# creatinf limits for the search space
bounds = np.zeros([Dim, 2])
bounds[:, 1] = np.ones(len(bounds))*10

# setting linear time delay between evaluations
timeDelay = .1

# setting the memory of the system
Mem = 50
# defining the kernel for the gaussian processes
kernel = Matern(length_scale=.7, nu=2.5)





if Dim == 2:
    # best and worse paths (for 2D)
    x = np.linspace(0, 10, 100)
    y = np.linspace(0, 1000, 1000)
    Xsearch = np.meshgrid(x, x, y)
    TotFunc = model(Xsearch, ThetaCritFuncPos, 8) - model(Xsearch, ThetaCritFuncNeg, 3)
    TotCont = model(Xsearch, ThetaControl, .1)
    MaxFunc = np.max(np.max(TotFunc * (TotCont < 0.05), axis=0), axis=0)
    MinFunc = np.min(np.min(TotFunc * (TotCont < 0.05), axis=0), axis=0)
    MaxCont = np.max(np.max(TotCont, axis=0), axis=0)
    MinCont = np.min(np.min(TotCont, axis=0), axis=0)
    for k in range(time):
        Amax[k] = np.unravel_index(np.argmax(TotFunc[:, :, k] * (TotCont[:, :, k] < 0.05)), TotFunc[:, :, k].shape)

# system starts here
for t in range(time):

    print('runing', t+1, ' / ', time)
    if t < 15:# first random points
        coord[0:Dim, t] = np.random.rand(Dim)*10
        coord[Dim, t] = t
        timeVect[0:t] += timeDelay

    else:
        if t < Mem: # before the memory buffer is full
            # getting the new coordinates to evaluate
            coord[0:Dim, t], risk[t] = Opt.GetNextOptimum(coord[0:Dim, 0:t].T, evalFunc[0:t], evalControl[0:t],
                                                 timeVect[0:t], .5, 0.5, 0.05, 100, 0.008, 100, kernel, 0.001, bounds, rtnRisk = True)
            # adding the time value to coordinate
            coord[Dim, 0:t] = t
            # updating the time since collection for every past evaluation
            timeVect[0:t] += timeDelay

        else: # once the memory buffer is full
            coord[0:Dim, t], risk[t] = Opt.GetNextOptimum(coord[0:Dim, t-Mem:t].T, evalFunc[t-Mem:t], evalControl[t-Mem:t],
                                                 timeVect[t-Mem:t], .5, 0.5, 0.05, 100, 0.008, 100, kernel, 0.001,
                                                 bounds, rtnRisk = True)
            coord[Dim, t] = t
            timeVect[0:t] += timeDelay

    # Query: evaluate the objective function and the control function at the new coordinates
    evalFunc[t] = model(coord[:, t], ThetaCritFuncPos, 8) - model(coord[:, t], ThetaCritFuncNeg, 3)
    evalControl[t] = model(coord[:, t], ThetaControl, .1)

    print('coords', coord[:, t])
    print('Objective', evalFunc[t])
    print('Control', evalControl[t])
    if t > 15:
        print('Objfail', np.sum(evalFunc[16:t+1] < 0.5))
        print('Confail', np.sum(evalControl[16:t+1] > 0.05))
        print('risk: ', risk[t])

if Dim == 2:

    plt.figure()
    plt.plot(MaxFunc, 'k--')
    plt.plot(evalFunc, 'k')
    plt.plot(np.ones(len(evalFunc))*.5, 'r')
    plt.plot(risk[:, 0], 'b')
    plt.plot(MinFunc, 'k--')
    plt.xlabel('time')
    plt.ylabel('Efficiency')
    plt.legend([r'min & max', r'Result', r'Func Threshold', r'estimated risk'])

    plt.figure()
    plt.plot(MaxCont, 'k--')
    plt.plot(evalControl, 'k')
    plt.plot(np.ones(len(evalFunc))*.05, 'r')
    plt.plot(risk[:, 1]/10, 'b')
    plt.plot(MinCont, 'k--')
    plt.xlabel('time')
    plt.ylabel('Control function')
    plt.legend([r'min & max', r'Result', r'Control Threshold', r'estimated risk / 10'])


    plt.figure()
    plt.subplot(211)
    plt.plot(coord[0])
    plt.plot(Amax[:, 1] / 10.)
    plt.xlabel('time')
    plt.ylabel('x1')
    plt.subplot(212)
    plt.plot(coord[1])
    plt.plot(Amax[:, 0] / 10.)
    plt.xlabel('time')
    plt.ylabel('x2')
    plt.title('Coordinate')
    plt.legend(['Estimator', 'Real coordinates'])

plt.show()






