import numpy as np
import matplotlib.pyplot as pl
import GPS_utils as u
import scipy.io.matlab as siom
from time import time as clock
from celerite import terms, GP
import emcee, sys, corner
import matplotlib.gridspec as gridspec

SPEED_OF_LIGHT = 2.99796458e8 # in m.s

# Load SNR=300 dataset
d = siom.loadmat('../data/synth_dataset_004_0300.mat')
wav = d['wavelength']
K, N = wav.shape
flux = d['flux']
flux_err = d['error']
baryvel = d['baryvel'].flatten()
lwav = np.log(wav * 1e-9)
dlw = baryvel / SPEED_OF_LIGHT
lwav_rest = lwav + dlw[:,None]
wav_rest = np.exp(lwav_rest) * 1e9

def fitspec(wav, flux, flux_err, nsteps = 2000, nrange = 3, prefix = 'RR1'):
    K, N = wav.shape
    lwav = np.log(wav * 1e-9) # in m
    lw0, lw1 = lwav.min(), lwav.max()
    x = (lwav - lw0) / (lw1 - lw0)
    # First do GP fit to one spectrum to get estimate of GP HPs
    i = np.random.randint(K)
    xx = x[i,:].flatten()
    yy = flux[i,:].flatten()
    ee = flux_err[i,:].flatten()
    xp = np.linspace(xx.min(), xx.max(), 100)
    HPs, _, _ = u.Fit0(xx, yy, ee, verbose = False, xpred = xp)
    print 'Initial GP HPs:', HPs
    # Initial (ML) estimate of parameters
    print "Starting ML fit"
    par_in = np.zeros(K+1)
    par_in[-2:] = HPs
    ML_par = np.array(u.Fit1(x, flux, flux_err, verbose = False, par_in = par_in))
    par_ML = np.copy(ML_par)
    par_ML[:K-1] *= (lw1 - lw0) * SPEED_OF_LIGHT
    par_ML[-1] *= (lw1 - lw0)
    print "ML fit done"
    # MCMC
    print "Starting MCMC"
    ndim = K+1
    nwalkers = ndim * 4
    p0 = ML_par + 1e-6 * np.random.randn(nwalkers, ndim)
    k = terms.Matern32Term(log_sigma = ML_par[-2], log_rho = ML_par[-1])
    gp = GP(k, mean = 1.0)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, u.LP1,
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
    samples_tpl[:,:,:K-1] *= (lw1 - lw0) * SPEED_OF_LIGHT
    samples_tpl[:,:,-1] *= (lw1 - lw0)
    par_MAP = np.copy(MAP_par)
    par_MAP[:K-1] *= (lw1 - lw0) * SPEED_OF_LIGHT
    par_MAP[-1] *= (lw1 - lw0)
    # parameter names for plots
    labels = []
    for i in range(K-1):
        labels.append(r'$\delta v_{%d}$ (m/s)' % (i+1))
    labels.append(r'$\ln \sigma$')
    labels.append(r'$\ln \rho$')
    labels = np.array(labels)
    names = []
    for i in range(K-1):
        names.append('dv_%d (m/s)' % (i+1))
    names.append('ln(sig)')
    names.append('ln(rho)')
    names = np.array(names)
    # Plot the chains
    fig1 = pl.figure(figsize = (12,K+3))
    gs1 = gridspec.GridSpec(ndim+1,1)
    gs1.update(left=0.1, right=0.98, bottom = 0.07, top = 0.98, hspace=0)
    ax1 = pl.subplot(gs1[0,0])
    axs = [ax1]
    pl.setp(ax1.get_xticklabels(), visible=False)
    pl.plot(Lprob.T, 'k-', alpha = 0.2)
    pl.ylabel(r'$\ln P$')
    for i in range(ndim):
        axc = pl.subplot(gs1[i+1,0], sharex = ax1)
        axs.append(axc)
        if i < (ndim-1):
            pl.setp(axc.get_xticklabels(), visible=False)
        pl.plot(samples_tpl[:,:,i].T, 'k-', alpha = 0.2)
        pl.ylabel(labels[i])
    pl.xlim(0,nsteps)
    pl.xlabel('iteration number')
    # Discard burnout
    nburn = int(0.25*nsteps)
    for ax in axs:
        ax.axvline(nburn)
    pl.savefig('../plots/%s_chains.png' % prefix)
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
    samples_flat = samples[:,nburn:,:].reshape(-1, ndim)
    samples_tpl_flat = samples_tpl[:,nburn:,:].reshape(-1, ndim)
    # Plot the parameter distributions
    fig2 = corner.corner(samples_tpl_flat, truths = par_MAP, labels = labels, show_titles = True, \
                            quantiles = [0.16, 0.84])
    pl.savefig('../plots/%s_corner.png' % prefix)
    return par_MAP, par50, par50-par16, par84-par50

k = 2
indices = np.arange(K)
ks = []
sigs = []
meds = []
ts = []
while k <= K:
    ks.append(k)
    print ''
    print '-----------------------'
    print 'Working with %d spectra' % k
    print '-----------------------'
    np.random.shuffle(indices)
    ii = indices[:k]
    print ii
    pl.close('all')
    t0 = clock()
    res = fitspec(wav_rest[ii,:], flux[ii,:], flux_err[ii,:], nsteps = 2000, prefix = 'RR1_%03d' % k)
    t1 = clock()
    ts.append(t1-t0)
    X = np.zeros((4,len(res[0])))
    X[0] = res[0]
    X[1] = res[1]
    X[2] = res[2]
    X[3] = res[3]
    np.savetxt('../data/test_RR1_%03d.dat' % k, X.T)
    print 'Time taken: %ds.' % int(ts[-1])
    meds.append(np.median(abs(res[1][:-2])))
    print 'Median absolute error: ', meds[-1]
    sigs.append(np.median(0.5*(res[2][:-2]+res[3][:-2])))
    print 'Median uncertainty: ', sigs[-1]
    k += 1
    X = np.zeros((4,len(ks)))
    X[0] = ks
    X[1] = ts
    X[2] = meds
    X[3] = sigs
    np.savetxt('../data/test_RR1_N.dat', X.T)
    raw_input('Next k?')
