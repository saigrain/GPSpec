import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from celerite import terms, GP

def NLL0(p, gp, y):
    gp.set_parameter_vector(p)
    g = gp.grad_log_likelihood(y, quiet=True)
    return -g[0], -g[1]

def Fit0(x, y, yerr = None, kernel = 'M32', HP_init = [0.0, 0.0], \
             mean = 1.0, verbose = True, doPlot = False, \
             xpred = None):
    if kernel == 'M32':
        k = terms.Matern32Term(log_sigma = HP_init[0], log_rho = HP_init[1])
    else:
        k = kernel(HP_init)
    gp = GP(k, mean = mean)
    gp.compute(x, yerr = yerr)
    soln = minimize(NLL0, HP_init, jac=True, args=(gp,y))
    gp.set_parameter_vector(soln.x)
    if verbose:
        print 'Initial pars:', HP_init
        print 'Fitted pars:', soln.x
    if xpred is None:
        xpred = x[:]
    mu, var = gp.predict(y, xpred, return_var = True)
    std = np.sqrt(var)
    if doPlot:
        plt.errorbar(xpred, y, yerr = yerr, fmt = ".k", capsize = 0)
        plt.plot(xpred, mu, 'C0')
        plt.fill_between(xpred, mu + std, mu - std, 'C0', alpha = 0.4, lw = 0)
    return soln.x, mu, std

def pairwise_ccf(spec, maxlags = None):
    K,N = spec.shape
    lags = np.arange(-N+1,N)
    ccfs = np.zeros((K,K,2*N-1))
    for i in np.arange(K):
        x = spec[i,:].flatten()
        for j in np.arange(K):
            if j > i:
                continue
            print i, j
            y = spec[j,:].flatten()
            ccfs[i,j,:] = np.correlate(x, y, 'full') / np.sqrt(np.dot(x, x) * np.dot(y, y))
    if maxlags:
        l = abs(lags) <= maxlags
        lags = lags[l]
        ccfs = ccfs[:,:,l]
    return lags, ccfs

def Wolkswagen(x2d, y2d, y2derr, truevel, doLog = True, sampling = 100, wav2si = 1.0e-10):
    K, N = x2d.shape
    if doLog:
        xang = x2d * wav2si
        log_xpred = np.linspace(np.log(xang.min()), np.log(xang.max()), N*sampling)
        xpred = np.exp(log_xpred) / wav2si
        dlw = log_xpred[1]-log_xpred[0]
    else:
        xpred = np.linspace(x2d.min(), x2d.max(), N*sampling)
        dw = xpred[1]-xpred[0]
    HPs = np.zeros((K,2))
    ypred = np.zeros((K,len(xpred)))
    plt.figure(1)
    plt.clf()
    for i in range(K):
        x = x2d[i,:].flatten()
        y = y2d[i,:].flatten()
        yerr = y2derr[i,:].flatten()
        plt.errorbar(x, y - i, yerr = yerr, fmt = 'k.', capsize = 0, alpha=0.5)
        res = Fit0(x, y, yerr, verbose = False, xpred = xpred)
        HPs[i,:] = res[0]
        ypred[i,:] = res[1]
        plt.plot(xpred, res[1] - i, color='C0')
        plt.fill_between(xpred, res[1] + res[2] - i, res[1] - res[2] - i, \
                             color = 'C0', alpha=0.4, lw=0)
    lags, ccfs = pairwise_ccf(ypred, 500)
    dvel = lags * dlw * 2.99796458e8 / 1e3
    plt.figure(2)
    plt.clf()
    for i in range(K):
        for j in range(K):
            if j > i: continue
            plt.clf()
            plt.plot(dvel, ccfs[i,j,:].flatten())
            ipk = np.argmax(ccfs[i,j,:].flatten())
            plt.axvline(dvel[ipk])
            print i,j, dvel[ipk], (truevel[j]-truevel[i])/1e3
            plt.draw()
            raw_input()

####################################################################################
# The CCF as implemented by correlate is not quite right, even as normalised here. #
# Probably best to focus on getting Rolls Royce working though next                #
####################################################################################



# def Fit0(x, y, yerr = None, kernel = 'M32', HP_init = [0.0, 0.0], \
#              mean = 1.0, verbose = True, doPlot = False, \
#              xpred = None):
             
# def NLL1(p, gp, x2d, y2d, y2derr):
#     x1d = (x2d + p[:-2, None]).flatten()
#     inds = np.argsort(x1d)
#     gp.set_parameter_vector(p[-2:])
#     y1d = y2d.flatten()
#     y1derr = y2derr.flatten()
#     gp.compute(x1d[inds], yerr = y1derr[inds])
#     g = gp.grad_log_likelihood(y1d[inds], quiet=True)
#     return -g[0], -g[1]

    
# def Fit1(x2d, y2d, y2derr = None, kernel = 'M32', HP_init = [0.0, 0.0], \
#              SH_init = None, mean = 1.0, verbose = True, doPlot = False):
#     if kernel == 'M32':
#         k = terms.Matern32Term()
#     else:
#         k = kernel()
#     gp = GP(k, mean = 1.0)
#     gp.compute(x, yerr = yerr)
#     if SH_init:
#         p0 = np.concatenate([SH_init, HP_init]).flatten()
#     else:
#         p0 = np.concatenate([np.zeros(K-1), HP_init]).flatten()
#     soln = minimize(NLL0, HP_init, jac=True)
#     gp.set_parameter_vector(soln.x)
#     mu, var = gp.predict(y, x, return_var = True)
#     std = np.sqrt(var)
#     if verbose:
#         print 'Initial pars:', HP_init
#         print 'Fitted pars:', soln.x
#     if doPlot:
#         plt.errorbar(x, y, yerr = yerr, fmt = ".k", capsize = 0)
#         plt.plot(x, mu, 'C0')
#         plt.fill_between(x, mu + std, mu - std, 'C0', alpha = 0.4, lw = 0)
#     return solnx, mu, std

# K, N = x2d.shape
#     np = len(p)
#     shifts = p[:K-1]
#     shifts = shift_func(shift_par, *shift_arg)
#     x2ds = np.copy(x2d)
    
#     if np == K+1
#     if np == K-1:
#         # fit shifts only
#         shifts = 
#     gp.set_parameter_vector(p)
#     g = gp.grad_log_likelihood(y, quiet=True)
#     return -g[0], -g[1]

# def Fit1Comp(x2d, y2d, y2derr = None, kernel = 'M32', HP_init = [0.0, 0.0], \
#                  mean = 1.0, XP_init = None, XP_func = None, XP_args = None, \
#                  verbose = True, doPlot = False):
#     K, N = x2d.shape
#     if kernel == 'M32':
#         k = terms.Matern32Term()
#     else:
#         k = kernel()
#     gp = GP(k, mean = 1.0)
#     gp.compute(x, yerr = yerr)
#     soln = minimize(NLL1, HP_init, jac=True)
#     gp.set_parameter_vector(soln.x)
#     mu, var = gp.predict(y, x, return_var = True)
#     std = np.sqrt(var)
#     if verbose:
#         print 'Initial pars:', HP_init
#         print 'Fitted pars:', soln.x
#     if doPlot:
#         plt.errorbar(x, y, yerr = yerr, fmt = ".k", capsize = 0)
#         plt.plot(x, mu, 'C0')
#         plt.fill_between(x, mu + std, mu - std, 'C0', alpha = 0.4, lw = 0)
#     return solnx, mu, std

