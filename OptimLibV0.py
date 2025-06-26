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
from statistics import NormalDist


class KFparAlt:
    def __init__(self,  cLin = 10E-3, cMeas = 10E+3):
        ''' Initialize:
        Create object with values for R and Q.
        X[0] with the first measure
        '''
        self.X = np.zeros(2) # X[0] contains the current estimate of the Beam mean X[1] is the estimate of the slope
        self.PX = np.zeros([2, 2]) # Covariance Matrix of X (Calculated by the filter)
        self.Sig = np.zeros(2) # Sig[0] contains the current estimate of Beam std
        self.PS = np.zeros([2, 2]) # Covariance Matrix of Sig (Calculated by the filter)
        self.Q = np.ones([2, 2]) * cLin  # relative confidence in the linear dynamic if increased less noise slower convergence
        self.R = np.array([[cMeas]])  # relative confidence in the measurement if increased faster convergence more noise
        self.F = np.array([[1., 1.], [0, 1.]])  # self dynamic of the system
        # [[link measure to state estimate, link fist order to state estimate],[link state to 1st order, propagate 1st order]]
        self.H = np.array([1., 0])  # link from the state space to the measures space (here the transformation from the measure to X[0] is 1)
        # and we do not measure directly dx/dt but deduce it with F
    #@classmethod
    def EstimateState(self, measure, deltaT):
        # extracting current values from the filter object

        PoldX = self.PX
        PoldSig = self.PS
        oldx = self.X
        oldSig = self.Sig

        F = self.F
        Q = self.Q
        R = self.R
        H = self.H
        F[0, 1] = deltaT # updating F

        # predictions
        xPred = np.dot(F,oldx)  # predicting the state xPred[k,0] = xEst[k-1,0] + (dx/dt)_estimate | xPred[k,1] = (dx/dt)_est
        pPred = np.dot(np.dot(F, PoldX),F.T) + Q  # Covariance matrix of the prediction (the bigger, the less confident we are)

        SigPred = np.dot(F, oldSig)  # Same thing but with standard deviation of the beam current
        SigpPred = np.dot(np.dot(F, PoldSig), F.T) + Q

        Inow = measure

        # updates

        y = Inow - np.dot(H,xPred)  # Calculating the innovation (diff between measure and prediction in the measure space)
        S = np.dot(np.dot(H, pPred),H.T) + R  # Calculating the Covariance of the measure (the bigger the less confident in the measure)

        K = np.dot(np.array([np.dot(PoldX, H.T)]).T, np.linalg.inv(S))  # Setting the Kalman optimal gain
        newX = xPred + np.dot(K, np.atleast_1d(y)).T  # Estimating the state at this instant
        PnewX = np.dot((np.eye(len(pPred)) - np.dot(K, np.array([H]))), pPred)  # Covariance matrix of the state

        # same steps followed for the standard deviation
        y = np.sqrt((Inow - newX[0]) ** 2) - np.dot(H, SigPred)  # Innovation of the standard deviation
        # this is an additional drawer to the Kalman filter and it is rather uncommon to estimate another variable that is
        # statistically dependent there may be better solutions, but this one works
        S = np.dot(np.dot(H, SigpPred), H.T) + R
        K = np.dot(np.array([np.dot(PoldSig, H.T)]).T, np.linalg.inv(S))
        newSig = (SigPred + np.dot(K, np.atleast_1d(y)).T)
        PnewSig = np.dot((np.eye(len(pPred)) - np.dot(K, np.array([H]))), SigpPred)

        # Updating the Kalman filter object
        self.PX = PnewX
        self.X = newX
        self.Sig = newSig
        self.PS = PnewSig





class KFpar:
    def __init__(self,  cLin = 10E-3, cMeas = 10E+3):
        ''' Initialize:
        Create object with values for R and Q.
        X[0] with the first measure
        '''
        self.X = np.zeros(2) # X[0] contains the current estimate of the Beam mean
        self.PX = np.zeros([2, 2]) # Covariance Matrix of X
        self.Sig = np.zeros(2) # Sig[0] contains the current estimate of Beam std
        self.PS = np.zeros([2, 2]) # Covariance Matrix of Sig
        self.Q = np.ones([2, 2]) * cLin  # relative confidence in the linear dynamic if increased less noise slower convergence
        self.R = np.array([[cMeas]])  # relative confidence in the measurement if increased faster convergence more noise
        self.F = np.array([[1., 1.], [0, 1.]])  # self dynamic of the system
        self.H = np.array([1., 0])  # link from the state space to the measures space




#Aquisition function, computes the Expected improvement for the input x
def EiComp(x, gp, y, nu):
    ''' Computes the expected improvement of x under the Gaussian process gp with the max value so far y and the exploration bias nu'''
    mu, sig = gp.predict(x, return_std=True)
    Ei = (mu - np.max(y) - nu) * scipy.stats.norm.cdf((mu - np.max(y) - nu)/sig) + sig * scipy.stats.norm.pdf((mu-np.max(y)- nu)/sig)
    return Ei

def buildRegressor(X, y, kernel, alpha):
    ''' return the Gaussian process regressor of X on y'''
    regY = GaussianProcessRegressor(kernel=kernel, alpha = alpha, random_state=0)
    regY.fit(X, y)

    return regY

def PredictCurrentValue(KFparam, measure, RegY = 0):
    ''' this function update the Kalman filter parameter including the prediction of the beam current
    and  the instablility with the measure of the proxy parameters and the regressor obtained during the
    tuning process
    KFparam is a KFpar type object containing the vectors for the Beam current estimate and the stability
    measure is a 1D vector containing the current values of the proxy parameters
    Regy id the regressor from the proxy to the instantaneous estimate of the beam current'''

    Q = KFparam.Q
    R = KFparam.R
    F = KFparam.F
    H = KFparam. H
    PoldX = KFparam.PX
    PoldSig = KFparam.PS
    oldx = KFparam.X
    oldSig = KFparam.Sig



    # predictions
    xPred = np.dot(F, oldx)
    pPred = np.dot(np.dot(F, PoldX), F.T) + Q



    SigPred = np.dot(F, oldSig)
    SigpPred = np.dot(np.dot(F, PoldSig), F.T) + Q

    if RegY:
        Inow = RegY.predict(np.asarray(measure).reshape(-1,1).T)
    else:
        Inow = measure

    # updates

    y = Inow - np.dot(H, xPred)  # ot used
    S = np.dot(np.dot(H, pPred), H.T) + R


    K = np.dot(np.array([np.dot(PoldX, H.T)]).T, np.linalg.inv(S))
    newX = xPred + np.dot(K, y).T
    PnewX = np.dot((np.eye(len(pPred)) - np.dot(K, np.array([H]))), pPred)



    y = np.sqrt((Inow - newX[0]) ** 2) - np.dot(H, SigPred)
    S = np.dot(np.dot(H, SigpPred), H.T) + R
    K = np.dot(np.array([np.dot(PoldSig, H.T)]).T, np.linalg.inv(S))
    newSig = (SigPred + np.dot(K, [y]).T)#[:, 0]
    PnewSig = np.dot((np.eye(len(pPred)) - np.dot(K, np.array([H]))), SigpPred)


    KFparam.PX = PnewX
    KFparam.X = newX
    KFparam.Sig = newSig[0]
    KFparam.PS = PnewSig

    return KFparam

def GetNextOptimum(X, y, c, t, Exp, OptThresh, ControlThresh, nMC, sigMC, limitMC, kernel, alpha, bounds, rtnRisk = False):

    ''' Feed the function with the last recorded values:
     X is a NxT matrix of the N inputs values for the past T time samples
     with T increases the Gaussian process complexity keep it relatively small
     the range of X have to be normalized and sigMC has to be consistent with the range
     y is a size T vector containing the associated output (beam current)
     c is a size T vector containing the control values (stability)
     t is a time vector containing a series of positive values increasing with how far in the
     past the corresponding X, y, and c were measured, t should increase faster from a sample to the previous
     if the system moves fast or if the two measures are separated by a longer time
     Exp is the Exploitation vs exploration trade off  for the Acquisition function
     OptThresh is the threshold under which we do not wish y
     ControlThresh is the threshold over which we do not wish c
     nMC is the number of Markov Chains generated to evaluate the Gaussian processes around the current functioning point
     sigMC is the standard deviation determining how far the next point can fall from the previous in the MC
     limitMC is the number of max iteration for each MC if it doesn't find an acceptable solution
     kernel and alpha are parameters of the gaussian processes
     bounds is a Nx2 matrix containing the limits of the space search for each parameter
     rtnRisk id a bool determining if the user wants to get the risks associated with the next parameter coordinates.
     '''

    risk = [0, 0]
    # building gaussian processes

    # Gaussian process for objective function
    GPY = GaussianProcessRegressor(kernel=kernel, alpha=alpha, random_state=0)
    GPY.fit(np.concatenate((X, np.asarray([t]).T), axis = 1), y)
    # Gaussian process for control function
    GPC = GaussianProcessRegressor(kernel=kernel, alpha=alpha, random_state=0)
    GPC.fit(np.concatenate((X, np.asarray([t]).T), axis = 1), c)

    Sol = []
    AQ = []
    # exploring the GPs with Markov chains
    for cptMC in range(nMC):
        run = True
        cptrun = 0
        # initializing on the current functioning point
        Pos = X[len(X)-1]
        # If system is currently both under control and over the objective threshold
        if (y[len(y)-1] > OptThresh) and (c[len(y)-1] < ControlThresh):
            while (run):
                # Generating the next particle
                Pos = Pos + np.random.randn(len(X[0])) * sigMC
                # Applying a bounce back if Pos gets out of the search bounds
                for k in range(len(Pos)):
                    if Pos[k] < bounds[k, 0]:
                        Pos[k] = bounds[k, 0] + abs(bounds[k, 0] - Pos[k])
                    if Pos[k] > bounds[k, 1]:
                        Pos[k] = bounds[k, 1] - abs(bounds[k, 1] - Pos[k])
                # Predicting the statistics of objective and control function at the new position
                GpredY, SpredY = GPY.predict(np.concatenate((Pos, np.asarray([0.]))).reshape(-1, 1).T, return_std=True)
                GpredC, SpredC = GPC.predict(np.concatenate((Pos, np.asarray([0.]))).reshape(-1, 1).T, return_std=True)
                # If the new position obey the risk mitigation constraints
                if ((GpredY - 2 * SpredY) > OptThresh) and ((GpredC + 2 * SpredC) < ControlThresh):
                    Sol.append(Pos)
                    AQ.append((EiComp(np.concatenate((Pos, np.asarray([0.]))).reshape(-1, 1).T, GPY, y, Exp)))
                else:
                    # If MC ends up in an area too risky
                    run = False
                if cptrun >= limitMC:
                    run = False
                cptrun += 1
        # If the system is currently out-of-control or under the objective threshold (can be optimized if too slow)
        else:
            # Random initialization of the search
            Pos = np.random.rand(len(X[0])) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
            while (run):
                Pos = Pos + np.random.randn(len(X[0])) * sigMC
                for k in range(len(Pos)):
                    if Pos[k] < bounds[k, 0]:
                        Pos[k] = bounds[k, 0] + abs(bounds[k, 0] - Pos[k])
                    if Pos[k] > bounds[k, 1]:
                        Pos[k] = bounds[k, 1] - abs(bounds[k, 1] - Pos[k])
                GpredY, SpredY = GPY.predict(np.concatenate((Pos, np.asarray([0.]))).reshape(-1, 1).T, return_std=True)
                GpredC, SpredC = GPC.predict(np.concatenate((Pos, np.asarray([0.]))).reshape(-1, 1).T, return_std=True)
                if ((GpredY - 2 * SpredY) > OptThresh) and ((GpredC + 2 * SpredC) < ControlThresh):
                    Sol.append(Pos)
                    AQ.append((EiComp(np.concatenate((Pos, np.asarray([0.]))).reshape(-1, 1).T, GPY, y, Exp)))
                if cptrun >= limitMC:
                    run = False
                cptrun += 1
    # if no solution has a low enough risk -> back to regular Bayesian Optimization
    if not len(Sol):
        print('No acceptable solution')
        # If we cannot find any solution that satisfies the constraints, switch to a standard BO
        for k in range(100):
            Pos = np.random.rand(len(X[0])) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
            Sol.append(Pos)
            AQ.append((EiComp(np.concatenate((Pos, np.asarray([0.]))).reshape(-1, 1).T, GPY, y, Exp * 3)))

    index = np.argmax(AQ)
    OptimX = Sol[index]

    if rtnRisk:
        # Evaluating the risks of the proposed solution
        GpredY, SpredY = GPY.predict(np.concatenate((OptimX, np.asarray([0.]))).reshape(-1, 1).T, return_std=True)
        GpredC, SpredC = GPC.predict(np.concatenate((OptimX, np.asarray([0.]))).reshape(-1, 1).T, return_std=True)

        risk[0] = NormalDist(mu=GpredY, sigma=SpredY).cdf(OptThresh) # may need double check
        risk[1] = 1 - NormalDist(mu=GpredC, sigma=SpredC).cdf(ControlThresh)


        return OptimX, risk

    else:

        return OptimX