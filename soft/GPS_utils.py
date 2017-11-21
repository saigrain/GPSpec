import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from celerite import terms, GP
import george
from scipy.sparse.linalg import LinearOperator, cg
import emcee, corner
import sys
import matplotlib.gridspec as gridspec
import scipy.io.matlab as siom
import scipy.linalg as sla

SPEED_OF_LIGHT = 2.99796458e8

####################################################################################
# Routines for fitting one spectrum at a time                                      #
####################################################################################

def NLL0(p, gp, y):
    gp.set_parameter_vector(p)
    g = gp.grad_log_likelihood(y, quiet=True)
    return -g[0], -g[1]

def Fit0(x, y, yerr, verbose = True, doPlot = False, \
             xpred = None):
    k = terms.Matern32Term(log_sigma = 0.0, log_rho = 0.0)
    gp = GP(k, mean = 1.0)
    gp.compute(x, yerr = yerr)
    soln = minimize(NLL0, gp.get_parameter_vector(), jac=True, args=(gp,y))
    gp.set_parameter_vector(soln.x)
    if verbose:
        print 'Initial pars:', HP_init
        print 'Fitted pars:', soln.x
    if xpred is None:
        return soln.x
    mu, var = gp.predict(y, xpred, return_var = True)
    std = np.sqrt(var)
    if doPlot:
        plt.errorbar(x, y, yerr = yerr, fmt = ".k", capsize = 0)
        plt.plot(xpred, mu, 'C0')
        plt.fill_between(xpred, mu + std, mu - std,  color = 'C0', alpha = 0.4, lw = 0)
    return soln.x, mu, std

def Fit0_Jitter(x, y, yerr, verbose = True, doPlot = False, \
                    xpred = None):
    k = terms.Matern32Term(log_sigma = 0.0, log_rho = 0.0)
    k += terms.JitterTerm(log_sigma = np.log(np.median(yerr)))
    gp = GP(k, mean = 1.0)
    gp.compute(x)
    soln = minimize(NLL0, gp.get_parameter_vector(), jac=True, args=(gp,y))
    gp.set_parameter_vector(soln.x)
    if verbose:
        print 'Initial pars:', HP_init
        print 'Fitted pars:', soln.x
    if xpred is None:
        return soln.x
    mu, var = gp.predict(y, xpred, return_var = True)
    std = np.sqrt(var)
    print xpred.shape
    print mu.shape
    print std.shape
    if doPlot:
        plt.errorbar(x, y, yerr = yerr, fmt = ".k", capsize = 0)
        plt.plot(xpred, mu, 'C0')
        plt.fill_between(xpred, mu + std, mu - std, color = 'C0', alpha = 0.4, lw = 0)
    return soln.x, mu, std

####################################################################################
# Single component case - using celerite out of the box                            #
####################################################################################

def LP1(p, gp, x2d, y2d, y2derr):
    K = x2d.shape[0]
    shifts = np.append(0, p[:K-1])
    x1d = (x2d + shifts[:, None]).flatten()
    inds = np.argsort(x1d)
    gp.set_parameter_vector(p[K-1:])
    y1d = y2d.flatten()
    y1derr = y2derr.flatten()
    try:
        gp.compute(x1d[inds], yerr = y1derr[inds])
    except LinAlgError:
        return -np.inf
    return gp.log_likelihood(y1d[inds], quiet=True)

def NLL1(p, gp, x2d, y2d, y2derr):
    return -LP1(p, gp, x2d, y2d, y2derr)

def Fit1(x2d, y2d, y2derr, par_in = None, verbose = True):
    K = x2d.shape[0]
    if par_in is None:
        par_in = np.zeros(K+1)        
    k = terms.Matern32Term(log_sigma = par_in[-2], log_rho = par_in[-1])
    gp = GP(k, mean = 1.0)
    soln = minimize(NLL1, par_in, args=(gp, x2d, y2d, y2derr))
    if verbose:
        print 'Initial pars:', par_in
        print 'Fitted pars:', soln.x
    return soln.x

def Pred1_2D(par, x2d, y2d, y2derr, doPlot = True, x2dpred = None):
    K = x2d.shape[0]
    k = terms.Matern32Term(log_sigma = par[-2], log_rho = par[-1])
    gp = GP(k, mean = 1.0)
    shifts = np.append(0,par[:K-1]) 
    x1d = (x2d + shifts[:, None]).flatten()
    inds = np.argsort(x1d)
    y1d = y2d.flatten()
    y1derr = y2derr.flatten()
    gp.compute(x1d[inds], yerr = y1derr[inds])
    if x2dpred is None:
        x2dpred = np.copy(x2d)
    x1dpred = (x2dpred + shifts[:, None]).flatten()
    indspred = np.argsort(x1dpred)
    mu, var = gp.predict(y1d[inds], x1dpred[indspred], return_var = True)
    std = np.sqrt(var)
    y1dpred = np.zeros_like(x1dpred)
    y1dpred[indspred] = mu
    y1dprederr = np.zeros_like(x1dpred)
    y1dprederr[indspred] = std
    y2dpred = y1dpred.reshape(x2dpred.shape)
    y2dprederr = y1dprederr.reshape(x2dpred.shape)    
    if doPlot:
        for i in range(K):
            plt.errorbar(x2d[i,:], y2d[i,:] - i, yerr = y2derr[i,:], fmt = ".k", capsize = 0, alpha = 0.5)
            plt.plot(x2dpred[i,:], y2dpred[i,:] - i, 'C0')
            plt.fill_between(x2dpred[i,:], y2dpred[i,:] + y2dprederr[i,:] - i, \
                                 y2dpred[i,:] - y2dprederr[i,:] - i, color = 'C0', alpha = 0.4, lw = 0)
    return x2dpred, y2dpred, y2dprederr

def GPSpec_1Comp(wav, flux, flux_err, nsteps = 2000, nrange = 3, prefix = 'RR1'):
    # NB: input wavelengths should be in nm, flux continuum should be about 1
    K, N = wav.shape
    # Create 2-D array of scaled log wavelengths for fitting
    lwav = np.log(wav * 1e-9) # in m
    lw0, lw1 = lwav.min(), lwav.max()
    x = (lwav - lw0) / (lw1 - lw0)
    # First do GP fit to individual spectra to get estimate of GP HPs
    print 'GP fit to individual spectra'
    HPs = np.zeros((K,2))
    for i in range(K):
        xx = x[i,:].flatten()
        yy = flux[i,:].flatten()
        ee = flux_err[i,:].flatten()
        HPs[i,:] = Fit0(xx, yy, ee, verbose = False, xpred = None)
    HPs = np.median(HPs, axis=0)
    print 'Initial GP HPs:', HPs
    # Initial (ML) estimate of parameters
    print "Starting ML fit"
    par_in = np.zeros(K+1)
    par_in[-2:] = HPs
    ML_par = np.array(Fit1(x, flux, flux_err, verbose = False, par_in = par_in))
    par_ML = np.copy(ML_par)
    par_ML[:K-1] *= (lw1 - lw0) * SPEED_OF_LIGHT * 1e-3
    par_ML[-1] *= (lw1 - lw0)
    k = terms.Matern32Term(log_sigma = ML_par[-2], log_rho = ML_par[-1])
    gp = GP(k, mean = 1.0)
    print "ML fit done"
    # MCMC
    print "Starting MCMC"
    ndim = K+1
    nwalkers = ndim * 4
    p0 = ML_par + 1e-4 * np.random.randn(nwalkers, ndim)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, LP1,
                                        args = [gp, x, flux, flux_err])
    for i, result in enumerate(sampler.sample(p0, iterations=nsteps)):
        n = int((30+1) * float(i) / nsteps)
        sys.stdout.write("\r[{0}{1}]".format('#' * n, ' ' * (30 - n)))
    sys.stdout.write("\n")
    print("MCMC done")
    # find MAP parameters
    iMAP = np.argmax(sampler.flatlnprobability)
    MAP_par = sampler.flatchain[iMAP,:].flatten()
    # extract MCMC chains
    samples = sampler.chain
    Lprob = sampler.lnprobability    
    # convert chains back to physical units: shifts in km/s
    samples_tpl = np.copy(samples)
    samples_tpl[:,:,:K-1] *= (lw1 - lw0) * SPEED_OF_LIGHT * 1e-3
    samples_tpl[:,:,-1] *= (lw1 - lw0)
    par_MAP = np.copy(MAP_par)
    par_MAP[:K-1] *= (lw1 - lw0) * SPEED_OF_LIGHT * 1e-3
    par_MAP[-1] *= (lw1 - lw0)
    # parameter names for plots
    labels = []
    for i in range(K-1):
        labels.append(r'$\delta v_{%d}$ (km/s)' % (i+1))
    labels.append(r'$\ln \sigma$')
    labels.append(r'$\ln \rho$')
    labels = np.array(labels)
    names = []
    for i in range(K-1):
        names.append('dv_%d (km/s)' % (i+1))
    names.append('ln(sig)')
    names.append('ln(rho)')
    names = np.array(names)
    # Plot the chains
    fig1 = plt.figure(figsize = (12,K+3))
    gs1 = gridspec.GridSpec(ndim+1,1)
    gs1.update(left=0.1, right=0.98, bottom = 0.07, top = 0.98, hspace=0)
    ax1 = plt.subplot(gs1[0,0])    
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.plot(Lprob.T, 'k-', alpha = 0.2)
    plt.ylabel(r'$\ln P$')
    for i in range(ndim):
        axc = plt.subplot(gs1[i+1,0], sharex = ax1)    
        if i < (ndim-1):
            plt.setp(axc.get_xticklabels(), visible=False)
        plt.plot(samples_tpl[:,:,i].T, 'k-', alpha = 0.2)
        plt.ylabel(labels[i])
    plt.xlim(0,nsteps)
    plt.xlabel('iteration number')
    # Discard burnout
    nburn = int(raw_input('Enter no. steps to discard as burnout: '))
    plt.axvline(nburn)
    # Evaluate and print the parameter ranges
    print '\n{:20s}: {:10s} {:10s} {:10s} - {:7s} + {:7s}'.format('Parameter', 'ML', 'MAP', \
                                                                    'Median','Error','Error')
    par50 = np.zeros(ndim)
    par84 = np.zeros(ndim)
    par16 = np.zeros(ndim)
    for i in range(ndim):
        sam = samples_tpl[:,:,i].flatten()
        b, m, f = np.percentile(sam, [16,50,84])
        par50[i] = m
        par16[i] = b
        par84[i] = f
        print '{:20s}: {:10.5f} {:10.5f} {:10.5f} - {:7.5f} + {:7.5f}'.format(names[i], \
                                                                                  par_ML[i], \
                                                                                  par_MAP[i], \
                                                                                  m, m-b, f-m)
    if prefix is None:
        return par_MAP, par50, par50-par16, par84-par50
    plt.savefig('%s_chains.png' % prefix)
    samples_flat = samples[:,nburn:,:].reshape(-1, ndim)
    samples_tpl_flat = samples_tpl[:,nburn:,:].reshape(-1, ndim)
    # Plot the parameter distributions
    fig2 = corner.corner(samples_tpl_flat, truths = par_MAP, labels = labels, show_titles = True, \
                            quantiles = [0.16, 0.84])
    plt.savefig('%s_corner.png' % prefix)
    # Plot the individual spectra with MAP fit
    xpred, fpred, fpred_err = Pred1_2D(MAP_par, x, flux, flux_err, doPlot = False)
    lwpred = (lw1-lw0) * xpred + lw0
    wpred = np.exp(lwpred) * 1e9
    fig3 = plt.figure(figsize = (12,K+1))
    gs3 = gridspec.GridSpec(K,1)
    gs3.update(left=0.1, right=0.98, bottom = 0.07, top = 0.98, hspace=0)
    for i in range(K):
        if i == 0:
            ax1 = plt.subplot(gs3[0,0])
        else:
            axc = plt.subplot(gs3[i,0], sharex=ax1, sharey=ax1)
        if i < (K-1):
            plt.setp(ax1.get_xticklabels(), visible=False)
        plt.errorbar(wav[i,:], flux[i,:], yerr = flux_err[i,:], \
                         fmt = ".k", ms = 2, mec = 'none', capsize = 0, alpha = 0.5)
        plt.plot(wpred[i,:], fpred[i,:], 'C0')
        plt.fill_between(wpred[i,:], fpred[i,:] + 2 * fpred_err[i,:], \
                             fpred[i,:] - fpred_err[i,:], color = 'C0', alpha = 0.4, lw = 0)
        plt.ylabel('spec. %d' % (i+1))
    plt.xlim(wav.min(), wav.max())
    plt.xlabel('wavelength (nm)')
    plt.savefig('%s_spectra.png' % prefix)
    # Plot the combined spectra with samples from MCMC chain
    shifts = np.append(0,MAP_par[:K-1])
    x1d = (x + shifts[:, None]).flatten()
    lw1d = (lw1-lw0) * x1d + lw0
    w1d = np.exp(lw1d) * 1e9
    y1d = flux.flatten()
    y1derr = flux_err.flatten()
    inds = np.argsort(x1d)
    gp.set_parameter_vector(MAP_par[-2:])
    gp.compute(x1d[inds], yerr = y1derr[inds])
    fig4 = plt.figure(figsize = (12,nrange+1))
    gs4 = gridspec.GridSpec(nrange,1)
    gs4.update(left=0.1, right=0.98, bottom = 0.07, top = 0.98, hspace=0.05)
    ws = w1d.min()
    wr = (w1d.max()-ws) / float(nrange)
    for i in range(nrange):
        if i == 0:
            ax1 = plt.subplot(gs4[0,0])
        else:
            axc = plt.subplot(gs4[i,0], sharey=ax1)
        if i < (nrange-1):
            plt.setp(ax1.get_xticklabels(), visible=False)
        wmin = ws + (i - 0.05) * wr
        wmax = ws + (i + 1.05) * wr
        l = (w1d >= wmin) * (w1d <= wmax)
        plt.errorbar(w1d[l], y1d[l], yerr = y1derr[l], fmt = ".k", capsize = 0, \
                         alpha = 0.5, ms = 2, mec='none')
        wpred = np.linspace(wmin, wmax, 1000)
        lwpred = np.log(wpred * 1e-9)
        xpred = (lwpred-lw0)/(lw1-lw0)
        isamp = np.random.randint(nsteps-nburn, size=10)
        for j in isamp:
            samp_params = samples_flat[j,:].flatten()
            samp_shifts = np.append(0, samp_params[:K-1])
            x1_samp = (x + samp_shifts[:, None]).flatten()
            inds_samp = np.argsort(x1_samp)
            k_samp = terms.Matern32Term(log_sigma=samp_params[-2],log_rho=samp_params[-1])
            gp_samp = GP(k_samp, mean=1.)
            gp_samp.compute(x1_samp[inds_samp], yerr = y1derr[inds_samp])
            mu, _ = gp.predict(y1d[inds_samp], xpred, return_var = True)
            plt.plot(wpred, mu, 'C0-', lw = 0.5, alpha = 0.5)
        plt.xlim(wmin, wmax)
        plt.ylabel('flux')
    plt.xlabel('wavelength (nm)')
    plt.savefig('%s_combined.png' % prefix)
    return par_MAP, par50, par50-par16, par84-par50, [fig1, fig2, fig3, fig4]
    
def test1():
    plt.close('all')
    d = siom.loadmat('../data/synth_dataset_002.mat')
    wav = d['wavelength'] * 0.1
    flux = d['flux']
    flux_err = d['error']
    baryvel = d['baryvel'].flatten()
    starvel = d['starvel'].flatten()
    true_shifts = (baryvel[1:] + starvel[1:]) - (baryvel[0] + starvel[0])
    print true_shifts * 1e-3
    figs = GPSpec_1Comp(wav, flux, flux_err, nsteps = 2000, prefix = 'synth2')
    return

####################################################################################
# Two component case - using Dan's CG trick                                        #
####################################################################################

def NLL2(params, gp1, gp2, x2d, y2d):
    K, N = x2d.shape
    s1 = np.append(0.0, params[:K-1])
    x1 = (x2d + s1[:, None]).flatten()
    inds1 = np.argsort(x1)
    x1 = x1[inds1]
    s2 = np.append(0.0, params[K-1:])
    x2 = (x2d + s2[:, None]).flatten()
    inds2 = np.argsort(x2)
    x2 = x2[inds2]
    y1 = y2d.flatten()
    # Define a custom "LinearOperator"
    def matvec(v):
        res = np.empty_like(v)
        res[inds1] = gp1.dot(v[inds1], x1, check_sorted=False)[:, 0]
        res[inds2] += gp2.dot(v[inds2], x2, check_sorted=False)[:, 0]
        return res
    op = LinearOperator((K*N, K*N), matvec=matvec)
    # Solve the system and compute the first term of the log likelihood
    soln = cg(op, y1, tol=0.01)
    value = 0.5 * np.dot(y1, soln[0])
    return value

def LP2(p, gp1, gp2, x2d, y2d):
    return -NLL2(p, gp1, gp2, x2d, y2d)

def Fit2(x2d, y2d, gp1, gp2, verbose = True, par_in = None):
    K = x2d.shape[0]
    if par_in is None:
        par_in = np.zeros(2*(K-1))
    nll_in = NLL2(par_in, gp1, gp2, x2d, y2d)
    if verbose:
        nll_in = NLL2(par_in, gp1, gp2, x2d, y2d)
        print 'Initial NLL:', nll_in
        print 'Initial par:', par_in
    soln = minimize(NLL2, par_in, args=(gp1, gp2, x2d, y2d))
    if verbose:
        print 'Final NLL:', soln.fun
        print 'Final pars:', soln.x
    return soln.x

def Pred2_2D(par, gp1, gp2, x2d, y2d, y2derr, xpred = None):
    K, N = x2d.shape
    s1 = np.append(0.0, par[:K-1])
    x1 = (x2d + s1[:, None]).flatten()
    K1 = gp1.get_matrix(x1)
    s2 = np.append(0.0, par[K-1:])
    x2 = (x2d + s2[:, None]).flatten()
    K2 = gp2.get_matrix(x2)
    Ktot = K1 + K2 + np.diag((y2derr.flatten())**2)
    L = sla.cho_factor(Ktot)
    y = y2d.flatten() - 1.0
    b = sla.cho_solve(L, y)
    if xpred is None:
        xpred = np.copy(x2d)
    x1pred = (xpred + s1[:, None]).flatten()
    K1s = gp1.get_matrix(x1pred, x1)
    x2pred = (xpred + s2[:, None]).flatten()
    K2s = gp2.get_matrix(x2pred, x2)
    Ks = K1s + K2s
    K1ss = gp1.get_matrix(x1pred)
    K2ss = gp2.get_matrix(x2pred)
    Kss = K1ss + K2ss
    mu1 = np.dot(K1s, b).reshape(xpred.shape)
    mu2 = np.dot(K2s, b).reshape(xpred.shape)
    mu = np.dot(Ks, b).reshape(xpred.shape)
    c = sla.cho_solve(L, Ks.T)
    cov = Kss - np.dot(Ks, c)
    var = np.diag(cov)
    std = np.sqrt(var).reshape(xpred.shape)
    return mu+1, std, mu1+1, mu2+1

def GPSpec_2Comp(wav, flux, flux_err, shifts_in = None, nsteps = 2000, nrange = 3, prefix = 'RR2'):
    # NB: input wavelengths should be in nm, flux continuum should be about 1
    K, N = wav.shape
    # Create 2-D array of scaled log wavelengths for fitting
    lwav = np.log(wav * 1e-9) # in m
    lw0, lw1 = lwav.min(), lwav.max()
    x = (lwav - lw0) / (lw1 - lw0)
    # First do GP fit to individual spectra to get estimate of GP HPs
    print 'GP fit to individual spectra'
    HPs = np.zeros((K,3))
    for i in range(K):
        xx = x[i,:].flatten()
        yy = flux[i,:].flatten()
        ee = flux_err[i,:].flatten()
        HPs[i,:] = Fit0_Jitter(xx, yy, ee, verbose = False)
    HPs = np.median(HPs, axis=0)
    print 'GP HPs:', HPs
    k = terms.Matern32Term(log_sigma = HPs[0], log_rho = HPs[1])
    k += terms.JitterTerm(log_sigma = HPs[2])
    gp1 = GP(k, mean = 1.0)
    gp2 = GP(k, mean = 1.0)
    # Initial (ML) estimate of parameters
    print "Starting ML fit"
    if shifts_in is None:
        shifts_in = np.zeros(2*(K-1))
    par_in = shifts_in / SPEED_OF_LIGHT / (lw1-lw0)
    # ML_par = np.array(Fit2(x, flux, gp1, gp2, verbose = False, par_in = par_in))
    ML_par = np.copy(par_in)
    par_ML = np.copy(ML_par)
    par_ML *= (lw1 - lw0) * SPEED_OF_LIGHT * 1e-3
    print "ML fit done"
    # MCMC
    print "Starting MCMC"
    ndim = len(ML_par)
    nwalkers = ndim * 4
    p0 = ML_par + 1e-4 * np.random.randn(nwalkers, ndim)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, LP2,
                                        args = [gp1, gp2, x, flux])
    for i, result in enumerate(sampler.sample(p0, iterations=nsteps)):
        n = int((30+1) * float(i) / nsteps)
        print i
        sys.stdout.write("\r[{0}{1}]".format('#' * n, ' ' * (30 - n)))
    sys.stdout.write("\n")
    print("MCMC done")
    # find MAP parameters
    iMAP = np.argmax(sampler.flatlnprobability)
    MAP_par = sampler.flatchain[iMAP,:].flatten()
    # extract MCMC chains
    samples = sampler.chain
    Lprob = sampler.lnprobability    
    # convert chains back to physical units: shifts in km/s
    samples_tpl = np.copy(samples)
    samples_tpl *= (lw1 - lw0) * SPEED_OF_LIGHT * 1e-3
    par_MAP = np.copy(MAP_par)
    par_MAP *= (lw1 - lw0) * SPEED_OF_LIGHT * 1e-3
    # parameter names for plots
    labels = []
    for i in range(K-1):
        labels.append(r'$\delta v^1_{%d}$ (km/s)' % (i+1))
    for i in range(K-1):
        labels.append(r'$\delta v^2_{%d}$ (km/s)' % (i+1))
    labels = np.array(labels)
    names = []
    for i in range(K-1):
        names.append('dv1_%d (km/s)' % (i+1))
    for i in range(K-1):
        names.append('dv2_%d (km/s)' % (i+1))
    names = np.array(names)
    # Plot the chains
    fig1 = plt.figure(figsize = (12,2*(K-1)+2))
    gs1 = gridspec.GridSpec(ndim+1,1)
    gs1.update(left=0.1, right=0.98, bottom = 0.07, top = 0.98, hspace=0)
    ax1 = plt.subplot(gs1[0,0])    
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.plot(Lprob.T, 'k-', alpha = 0.2)
    plt.ylabel(r'$\ln P$')
    for i in range(ndim):
        print i, ndim, len(labels)
        axc = plt.subplot(gs1[i+1,0], sharex = ax1)    
        if i < (ndim-1):
            plt.setp(axc.get_xticklabels(), visible=False)
        plt.plot(samples_tpl[:,:,i].T, 'k-', alpha = 0.2)
        plt.ylabel(labels[i])
    plt.xlim(0,nsteps)
    plt.xlabel('iteration number')
    # Discard burnout
    # nburn = int(raw_input('Enter no. steps to discard as burnout: '))
    nburn = 0
    plt.axvline(nburn)
    # Evaluate and print the parameter ranges
    print '\n{:20s}: {:10s} {:10s} {:10s} - {:7s} + {:7s}'.format('Parameter', 'ML', 'MAP', \
                                                                    'Median','Error','Error')
    par50 = np.zeros(ndim)
    par84 = np.zeros(ndim)
    par16 = np.zeros(ndim)
    for i in range(ndim):
        sam = samples_tpl[:,:,i].flatten()
        b, m, f = np.percentile(sam, [16,50,84])
        par50[i] = m
        par16[i] = b
        par84[i] = f
        print '{:20s}: {:10.5f} {:10.5f} {:10.5f} - {:7.5f} + {:7.5f}'.format(names[i], \
                                                                                  par_ML[i], \
                                                                                  par_MAP[i], \
                                                                                  m, m-b, f-m)
    if prefix is None:
        return par_MAP, par50, par50-par16, par84-par50
    plt.savefig('%s_chains.png' % prefix)
    samples_flat = samples[:,nburn:,:].reshape(-1, ndim)
    samples_tpl_flat = samples_tpl[:,nburn:,:].reshape(-1, ndim)
    # Plot the parameter distributions
    fig2 = corner.corner(samples_tpl_flat, truths = par_MAP, labels = labels, show_titles = True, \
                            quantiles = [0.16, 0.84])
    plt.savefig('%s_corner.png' % prefix)
    # Plot the individual spectra with MAP fit
    xpred = np.copy(x)
    fpred, fpred_err, f1pred, f2pred = Pred2_2D(MAP_par, gp1, gp2, x, flux, flux_err, xpred = xpred)
    lwpred = (lw1-lw0) * xpred + lw0
    wpred = np.exp(lwpred) * 1e9
    fig3 = plt.figure(figsize = (12,K+1))
    gs3 = gridspec.GridSpec(K,1)
    gs3.update(left=0.1, right=0.98, bottom = 0.07, top = 0.98, hspace=0)
    for i in range(K):
        if i == 0:
            ax1 = plt.subplot(gs3[0,0])
        else:
            axc = plt.subplot(gs3[i,0], sharex=ax1, sharey=ax1)
        if i < (K-1):
            plt.setp(ax1.get_xticklabels(), visible=False)
        plt.plot(wpred[i,:], f1pred[i,:], 'C1')
        plt.plot(wpred[i,:], f2pred[i,:], 'C2')
        plt.fill_between(wpred[i,:], fpred[i,:] + 2 * fpred_err[i,:], \
                             fpred[i,:] - fpred_err[i,:], color = 'C0', alpha = 0.4, lw = 0)
        plt.plot(wpred[i,:], fpred[i,:], 'C0')
        plt.ylabel('spec. %d' % (i+1))
        plt.errorbar(wav[i,:], flux[i,:], yerr = flux_err[i,:], \
                         fmt = ".k", ms = 3, mec = 'none', capsize = 0, alpha = 0.5, lw=0.5)

        plt.xlim(wav.min(), wav.max())
    plt.xlabel('wavelength (nm)')
    plt.savefig('%s_spectra.png' % prefix)
    # Plot the combined spectra with samples from MCMC chain
    s1 = np.append(0.0, MAP_par[:K-1])
    x11d = (x + s1[:, None]).flatten()
    lw11d = (lw1-lw0) * x11d + lw0
    w11d = np.exp(lw11d) * 1e9
    K1 = gp1.get_matrix(x11d)
    s2 = np.append(0.0, MAP_par[K-1:])
    x21d = (x + s2[:, None]).flatten()
    lw21d = (lw1-lw0) * x21d + lw0
    w21d = np.exp(lw21d) * 1e9
    K2 = gp2.get_matrix(x21d)
    y1derr = flux_err.flatten()
    Ktot = K1 + K2 + np.diag(y1derr**2)
    y1d = flux.flatten() - 1.0
    L = sla.cho_factor(Ktot)
    b = sla.cho_solve(L, y1d)
    fig4 = plt.figure(figsize = (12,nrange+1))
    gs4 = gridspec.GridSpec(nrange,1)
    gs4.update(left=0.1, right=0.98, bottom = 0.07, top = 0.98, hspace=0.05)
    ws = min(w11d.min(), w21d.min())
    wr = (max(w11d.max(),w21d.max())-ws) / float(nrange)
    for i in range(nrange):
        if i == 0:
            ax1 = plt.subplot(gs4[0,0])
        else:
            axc = plt.subplot(gs4[i,0], sharey=ax1)
        if i < (nrange-1):
            plt.setp(ax1.get_xticklabels(), visible=False)
        wmin = ws + (i - 0.05) * wr
        wmax = ws + (i + 1.05) * wr
        l = (w11d >= wmin) * (w11d <= wmax)
        plt.errorbar(w11d[l], y1d[l], yerr = y1derr[l], fmt = ".k", capsize = 0, \
                         alpha = 0.5, ms = 2, mec='none')
        l = (w21d >= wmin) * (w21d <= wmax)
        plt.errorbar(w21d[l], y2d[l] - 1, yerr = y1derr[l], fmt = ".k", capsize = 0, \
                         alpha = 0.5, ms = 2, mec='none')
        wpred = np.linspace(wmin, wmax, 1000)
        lwpred = np.log(wpred * 1e-9)
        xpred = (lwpred-lw0)/(lw1-lw0)
        isamp = np.random.randint(nsteps-nburn, size=10)
        for j in isamp:
            samp_params = samples_flat[j,:].flatten()
            s1 = samp_params[:K-1]
            x1pred = (xpred + s1[:, None]).flatten()
            lw1pred = (lw1-lw0) * x1pred + lw0
            w1pred = np.exp(lw1pred) * 1e9
            K1s = gp1.get_matrix(x1pred, x1)
            s2 = samp_params[K-1:]
            x2pred = (xpred + s2[:, None]).flatten()
            lw2pred = (lw1-lw0) * x2pred + lw0
            w2pred = np.exp(lw2pred) * 1e9
            K2s = gp2.get_matrix(x2pred, x2)
            Ks = K1s + K2s
            K1ss = gp1.get_matrix(x1pred)
            K2ss = gp2.get_matrix(x2pred)
            Kss = K1ss + K2ss
            mu1 = np.dot(K1s, b).reshape(xpred.shape)
            mu2 = np.dot(K2s, b).reshape(xpred.shape)
            plt.plot(w1pred, mu1, 'C0-', lw = 0.5, alpha = 0.5)
            plt.plot(w2pred, mu2-1, 'C1-', lw = 0.5, alpha = 0.5)
        plt.xlim(wmin, wmax)
        plt.ylabel('flux')
    plt.xlabel('wavelength (nm)')
    plt.savefig('%s_combined.png' % prefix)
    return par_MAP, par50, par50-par16, par84-par50, [fig1, fig2, fig3, fig4]

def test2():
    plt.close('all')
    d = siom.loadmat('../data/synth_dataset_003.mat')
    wav = d['wavelength'] * 0.1
    K,N = wav.shape
    flux = d['flux']
    flux_err = d['error']
    baryvel = d['baryvel'].flatten()
    starvel = d['starvel']
    shifts = np.zeros(2*(K-1))
    s1 = baryvel + starvel[:,0].flatten()
    shifts[:K-1] = s1[1:]-s1[0]
    s2 = baryvel + starvel[:,1].flatten()
    shifts[K-1:] = s2[1:]-s2[0]
    print shifts / 1e3
    res = GPSpec_2Comp(wav, flux, flux_err, shifts_in = shifts, nsteps = 100, prefix = 'synth3')
    return

####################################################################################
# The code below is my crappy implementation of Vinesh's approach for the single   #
# component case. However, it doesn't quite work as expected and I've not pursued  #
# it further.                                                                      #
####################################################################################


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

