#coding = utf8!

#import cv2
#import glob
import numpy as np
import scipy
from scipy import stats
import matplotlib.pyplot as plt
#import scipy.optimize as opt
#import pickle
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF, Matern
from skimage import data, filters, color, morphology
from skimage.segmentation import flood, flood_fill


import warnings
warnings.filterwarnings("ignore")

plt.rcParams["figure.figsize"] = (20, 9)

#################
# Adaptive Bayesian Optimizer for Tracking 1D
#
#
#
#
#
#
#
#
#
#
#
#################

#Function used to simulate beam statistics
def model(coord, tet, amp = 1):
    d = 0
    x = coord[0]
    y = coord[1]
    for k in range(len(tet)):
        d += np.exp(-(((x - tet[k, 0])/(tet[k, 2])))**2 - ((y - tet[k, 1])/(tet[k, 3]))**2)

    return d

#Function to determine the search area of acceptable solutions
def makeMask(loc, GP, Sig, tr):

    mask = np.zeros(len(GP))
    mask[loc[0]] = 1
    for k in np.arange(loc[0]+1, len(GP)-1):
        if (mask[k-1] == 1) and ((GP[k] - 2 * Sig[k]) > tr):
            mask[k] = 1
    for k in np.arange(loc[0]-1, 0, -1):
        if (mask[k+1] == 1) and ((GP[k] - 2 * Sig[k]) > tr):
            mask[k] = 1

    #Matrix = ((GP - 2 * Sig) > tr) * 255
    #mask = ((GP - 2 * Sig) > tr)
    #mask = flood_fill(Matrix, (loc[0]), 1, tolerance=1) > 0

    return mask

#Aquisition function, computes the Expected improvement for the input x
def EiComp(x, gp, y, nu):

    mu, sig = gp.predict(x, return_std=True)
    Ei = (mu - np.max(y) - nu) * scipy.stats.norm.cdf((mu - np.max(y) - nu)/sig) + sig * scipy.stats.norm.pdf((mu-np.max(y)- nu)/sig)
    return Ei


######## Parameters tuning #############################################################################################
np.random.seed(8)
#Exploitation/Exploration parameters
Exp1stOpt = 1.5# Very exploratory
ExpAllOk = -.5# Not very exploratory
ExpNoSol = 5.# A little exploratory

#parameters of the function at the begining [amaxX, amaxY, stdX, stdY, A]
ThetaCritFuncPos = np.zeros([5, 4]) #Parametwers of <Ib>
ThetaCritFuncPos[:, 0] = np.random.rand(5) * 8 + 1 # x peak loc
ThetaCritFuncPos[:, 1] = np.random.rand(5) * 800 + 100 # t peak loc
ThetaCritFuncPos[0,1] = 0.
ThetaCritFuncPos[:, 2] = np.random.rand(5) * 3 + 1# x peak sig
ThetaCritFuncPos[:, 3] = np.random.rand(5) * 150 + 10 # x peak sig
ThetaCritFuncNeg = np.zeros([5, 4]) #Parametwers of <Ib>
ThetaCritFuncNeg[:, 0] = np.random.rand(5) * 10 # x peak loc
ThetaCritFuncNeg[:, 1] = np.random.rand(5) * 1000 # t peak loc
ThetaCritFuncNeg[:, 2] = np.random.rand(5) * 1 + 1 # x peak sig
ThetaCritFuncNeg[:, 3] = np.random.rand(5) * 80 + 10 # x peak sig
ThetaControl = np.zeros([3, 4]) #Parameters of \sig_{Ib}
ThetaControl[:, 0] = np.random.rand(3) * 10 # x peak loc
ThetaControl[:, 1] = np.random.rand(3) * 800 + 200 # x peak loc
ThetaControl[:, 2] = np.random.rand(3) * 1 + 1
ThetaControl[:, 3] = np.random.rand(3) * 50 + 30

sigNoise = .01 #Noise of measure (Uncertainty regarding the estimate for the beam current stats)
nit = 15 #Number of iterations to optimize the function before starting the experiment
rdit = 15#Number of random evaluations to start GPopt
DB = np.zeros([rdit+nit, 2])#Holds the coordinates where the function is evaluated and the time since they were [coords, time]
Mod = np.zeros(rdit+nit)#Value of the function evaluated at the coordinates of DB
Mod2 = np.zeros(rdit+nit)#Value of the stability function evaluated at the coordinates of DB

TimeFading = .1#Time shift between a sample and the next (Quite sensitive)
thresh = -.1 #Threshold under which the value of the beam current shouldn't be
stab = .04#Stability threshold (-20%)
N = 1000#Number of iterations

critical = 0
######## Code starts here ##############################################################################################

# Setting up the random eval of the function
for k in range(rdit):
    DB[k, 0] = np.random.rand() * 10
    DB[k, 1] = k * TimeFading
    Mod[k] = model([DB[k, 0], k], ThetaCritFuncPos) - model([DB[k, 0], k], ThetaCritFuncNeg, .5, ) + np.random.randn()*sigNoise
    Mod2[k] = model([DB[k, 0], k], ThetaControl) + np.random.randn()*sigNoise


next_p = []
#setting up the Gaussian process
kernel = .5 * Matern(length_scale=.1, nu=2.5)#
#kernel = 1.0 * Matern(length_scale=1., nu=1.5)#
kernel2 = .5 * Matern(length_scale=.1, nu=2.5)#
gpr = GaussianProcessRegressor(kernel=kernel, random_state=0)
gpr2 = GaussianProcessRegressor(kernel=kernel2, random_state=0)

#First optimization on static system
for k in range(nit):
    DB[:, 1] += TimeFading
    gpr.fit(DB[0:rdit+k], Mod[0:rdit+k])
    gpr2.fit(DB[0:rdit+k], Mod2[0:rdit+k])

    EI = np.zeros([100])
    coordVal = [np.linspace(0, 10, 100)]
    for k1 in range(100):

        EI[k1] = EiComp(np.asarray([coordVal[0][k1], 0.]).reshape(-1, 1).T, gpr, Mod, Exp1stOpt)
        eval = gpr2.predict(np.asarray([coordVal[0][k1], 0.]).reshape(-1, 1).T)
        if eval > stab:
            EI[k1] = 0

    next_p = np.where(EI == np.max(EI))
    DB[15 + k, 0] = coordVal[0][next_p[0][0]]
    Mod[15 + k] = model([DB[15 + k, 0], 15 + k], ThetaCritFuncPos) - model([DB[15 + k, 0], 15 + k], ThetaCritFuncNeg, .5) + np.random.randn() * sigNoise
    Mod2[15 + k] = model([DB[15 + k, 0], 15 + k], ThetaControl) + np.random.randn() * sigNoise

# Here the function is optimized and approximated with the GP



###################### Switching to a dynamic system

rndReset = 0

# init time label for GP

CurrentIndex = np.argmax(Mod)
CurrentPoint = DB[CurrentIndex, 0]

PosEst = np.zeros([N+1, 2])# stores the estimated value of the position of the max
Val = np.zeros(N+1)# Value of the function evaluated at PosEst
ValIns = np.zeros(N+1)# Value of the function evaluated at PosEst
VarFail = np.zeros(N+1)# Value of the function evaluated at PosEst
rndReset = np.zeros(N+1)
#init

PosEst[0, 0] = CurrentPoint
Val[0] = Mod[CurrentIndex]
ValIns[0] = Mod2[CurrentIndex]
bounds = [0, 10]# Global searching area
for k in range(N):
    print('iteration:', k+1, '/', N)
    # crit func move
    DB[:, len(DB[0])-1] += TimeFading # updating the time parameter. This sets the time distance standard the higher the least influencial the past is

    # Time window to limit computational complexity
    if k <= 50:
        gpr.fit(DB, Mod)
        XControl = DB.copy()
        XControl[:, 1] = XControl[:, 1] * .5
        gpr2.fit(XControl, Mod2)
    else:
        gpr.fit(DB[len(DB)-51:len(DB)-1], Mod[len(DB)-51:len(DB)-1])
        XControl = DB[len(DB)-51:len(DB)-1].copy()
        XControl[:, 1] = XControl[:, 1] * .5
        gpr2.fit(XControl, Mod2[len(DB)-51:len(DB)-1])


    #Grid evaluations of the GP mean and std to determine a 2D mask of acceptable solutions
    if 0:#Mod[len(Mod)-1] > thresh and Mod2[len(Mod)-1] < stab:
        lb = max([0, CurrentPoint - .5])
        ub = min([10, CurrentPoint + .5])
    else:
        lb = 0
        ub = 10
    X = np.linspace(lb, ub, 100)
    GP = np.zeros(100)
    GP2 = np.zeros(100)
    Sig = np.zeros(100)
    Sig2 = np.zeros(100)

    #Calculation of the GPs over the global search space
    for l1 in range(100):
        GP[l1], Sig[l1] = gpr.predict(np.asarray([X[l1], 0.]).reshape(-1, 1).T, return_std=True)
        GP2[l1], Sig2[l1] = gpr2.predict(np.asarray([X[l1], 0.]).reshape(-1, 1).T, return_std=True)

    #Mask of the search area (continuous area where the prob of getting under thresh is 5% at max and prob of getting unstable is < 5%)

    loc = [np.argmin(abs(np.linspace(lb, ub, 100) - CurrentPoint))]
    Mask = makeMask(loc, GP, Sig, thresh)
    if np.sum(Mask)==0:
        Mask = np.ones(np.shape(Mask))
    #Mask = GP - 2* Sig > thresh
    Mask2 = makeMask(loc, -GP2, -Sig2, -stab)
    if np.sum(Mask2):
        Mask = Mask * Mask2
    #Mask[np.where(((GP2 + 2 * Sig2)) > stab)] = 0
    #print(GP2+2 *Sig2)
    if np.sum(Mask) < 3:
        VarFail[k] = 1


    #If the mask provides no acceptable solution
    if VarFail[k]:

        #Get to the most stable within acceptable unstable area
        #Mask = makeMask(loc, GP, 0. * np.ones(len(Sig)), thresh)
        # GP2 = GP2 * Mask
        #Mask[np.where(((GP2 + 0 * Sig2)) > stab)] = 0
        #Mask = - makeMask(loc, GP2, Sig2, stab)

        #GP = GP * Mask

        #Mask[np.where((GP - 0 * Sig) < (thresh))] = 0

        #if np.sum(Mask) < 2:
        Mask = np.ones(len(GP))

        tmp = np.zeros(np.shape(X))
        for l1 in range(len(tmp)):
            tmp[l1] = (EiComp(np.asarray([X[l1], 0.]).reshape(-1, 1).T, gpr, Mod, ExpNoSol))
            #tmp[l1] = (EiComp(np.asarray([X[l1], 0.]).reshape(-1, 1).T, gpr, Mod, ExpNoSol)) - 0.5 * (EiComp(np.asarray([X[l1], 0.]).reshape(-1, 1).T, gpr2, Mod2, ExpNoSol))
        #tmp = GP - 1 * GP2

        #plt.figure()
        #plt.subplot(211)
        #plt.plot(tmp)
        #plt.subplot(212)
        #plt.plot(tmp * Mask)
        #plt.show()
        #if np.sum(tmp):
        newCoord = np.asarray(np.uint(np.unravel_index(np.argmax(tmp, axis=None), tmp.shape)))
        #else:
        #    newCoord[0] = np.random.rand() * 100.
        #    rndReset += 1
        #newCoord[0] = np.random.rand() * 100.
    else:
        tmp = np.zeros(np.shape(X))
        for l1 in range(len(tmp)):
            tmp[l1] = (EiComp(np.asarray([X[l1], 0.]).reshape(-1, 1).T, gpr, Mod, ExpAllOk))

        if np.sum(tmp):
            newCoord = np.asarray(np.unravel_index(np.argmax(tmp * Mask, axis=None), tmp.shape))
        else:
            newCoord[0] = np.random.rand() * 100.
            rndReset[k] = 1



    #if critical:
    #    rndReset+=1
    #    newCoord[0] = np.random.rand() * 100.
    #if newCoord[0] == 0.:
    #    rndReset+=1
    #    newCoord[0] = np.random.rand() * 100.

    # Measuring the statistics at the new point and saving the coordinates
    DB = np.concatenate((DB, [[np.linspace(lb, ub, 100)[newCoord[0]], 0.]]))#Coordinates

    Mod = np.concatenate((Mod, [model([DB[len(DB)-1, 0], k], ThetaCritFuncPos)-model([DB[len(DB)-1, 0], k], ThetaCritFuncNeg, .5) + sigNoise * np.random.randn()]))#Mean
    Mod2 = np.concatenate((Mod2, [model([DB[len(DB)-1, 0], k], ThetaControl) + sigNoise * np.random.randn()]))#Std

    # Updating working point and Saving Data
    CurrentIndex = len(DB) - 1
    CurrentPoint = DB[CurrentIndex, 0]
    PosEst[k + 1] = [CurrentPoint, k]
    Val[k + 1] = Mod[CurrentIndex]
    ValIns[k + 1] = Mod2[CurrentIndex]
    if Val[k+1] < thresh:
        critical = 1
    elif ValIns[k+1] > (stab + 0.1):
        critical = 1
    else:
        critical = 0
    print([np.linspace(0, 10, 100)[newCoord[0]]], Val[k + 1], np.sum(Mask), VarFail[k], critical)
    if 0:
        fig = plt.figure()
        ax = fig.add_subplot(311)
        ax.plot(GP)
        #ax.plot(GP * Mask)
        ax.scatter(DB[len(DB)-1, 0]*10, Mod[len(DB)-1])
        ax = fig.add_subplot(312)
        ax.plot(GP2)
        #ax.plot(GP2 * Mask)
        ax = fig.add_subplot(313)
        ax.plot(tmp)
        #ax.plot(tmp * Mask)

        plt.show()




X, Y = np.meshgrid(np.linspace(0,10,100), np.linspace(PosEst[0, 1], PosEst[len(PosEst)-1, 1], 1000))
z = model([X, Y], ThetaCritFuncPos) - model([X, Y], ThetaCritFuncNeg, .5)
z2 = model([X, Y], ThetaControl)

print('Efficiency:', np.sum(Val * (Val > thresh) * (ValIns < stab+0.1)) / np.sum((np.max(z, axis = 1) * (Val[0:len(Val)-1] > thresh) * (ValIns[0:len(Val)-1] < stab+0.1))) * (1- np.sum(((Val < thresh) + (ValIns > (stab+0.1))) >= 1)/len(Val)))
print('Fail rate:', np.sum(((Val < thresh) + (ValIns > (stab+0.1))) >= 1)/len(Val))
print('Control fail:', np.sum(ValIns > (stab +0.1))/len(ValIns))
print('Optim fail:', np.sum(Val < thresh)/len(Val))
print('Random reset:', np.sum(rndReset))
print('No solution:', np.sum(VarFail))

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.plot_surface(X, Y, z)
ax.scatter(PosEst[:, 0], PosEst[:, 1], Val, color = 'r')

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.plot_surface(X, Y, z2)
ax.scatter(PosEst[:, 0], PosEst[:, 1], ValIns, color = 'r')


X3 = X.copy()
Y3 = Y.copy()
z3 = z.copy()

X3[np.where(z < thresh)] = np.nan
X3[np.where(z2 > (stab+.1))] = np.nan

Y3[np.where(z < thresh)] = np.nan
Y3[np.where(z2 > (stab+.1))] = np.nan

z3[np.where(z < thresh)] = np.nan
z3[np.where(z2 > (stab+.1))] = np.nan


fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
#ax = fig.add_subplot(111)
#ax.plot_surface(X3, Y3, z3, cmap ='cool')
ax.contour(X3, Y3, z3, 100, cmap ='cool')
ax.scatter(PosEst[:, 0], PosEst[:, 1], Val, color = 'r')
#ax.scatter(PosEst[:, 0], PosEst[:, 1], color = 'r')
ax.set_xlabel(r"$\theta$")
ax.set_ylabel(r'$t$')
ax.set_zlabel(r'$f(\theta, t)$')


fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
#ax = fig.add_subplot(111)
ax.plot_surface(X3, Y3, z3, cmap ='cool')
#ax.contour(X3, Y3, z3, 100, cmap ='cool')
ax.scatter(PosEst[:, 0], PosEst[:, 1], Val, color = 'r')
#ax.scatter(PosEst[:, 0], PosEst[:, 1], color = 'r')
ax.set_xlabel(r"$\theta$")
ax.set_ylabel(r'$t$')
ax.set_zlabel(r'$f(\theta, t)$')
#ax.legend(['Real function', 'Running points'])

BestVal = np.zeros(len(Val))
for k in range(len(Val)-1):
    BestVal[k+1] = np.nanmax(z3[k])
BestVal[0] = Val[0]

plt.figure()
plt.plot(Val)
plt.plot(BestVal)
plt.legend(['Runing value','Best stable'])
#print('efficiency:', np.mean((BestVal[np.where(ValIns < (stab +0.1))]-Val[np.where(ValIns < (stab +0.1))])))

import pickle
pickle.dump([PosEst, Val, X3, Y3, z3], open('AIO8.dat', 'wb'))


plt.show()






