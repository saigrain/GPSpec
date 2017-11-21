import numpy as np
import pylab as pl
import glob
import george, emcee, corner
from time import time as clock
import scipy.io as sio

SPEED_OF_LIGHT = 2.99796458e8
nmax = 1000

###################
# read in spectra #
###################
datadir = '../data/'
d = sio.loadmat('%s/synth_dataset_001.mat' % datadir)
wav = d['wavelength']
flux = d['flux'] + 1
err = d['error']

###############################
# work on subset only for now #
###############################
print 'Selecting subset'
ifl = np.arange(35).astype(int)
nfl = len(ifl)
wav = wav[ifl,:]
flux = flux[ifl,:]
err = err[ifl,:]
npix = len(wav[0,:].flatten())

baryvel = np.zeros(nfl)
wav_corr = np.copy(wav)
lwav = np.log(wav*1e-10)
dlw = baryvel / SPEED_OF_LIGHT
lwav_corr = np.copy(lwav)

##################################
# GP optimization for 1 spectrum #
##################################

# x1 = wav[0,:].flatten()
# y1 = flux[0,:].flatten()
# e1 = err[0,:].flatten()
# a0 = 0.19
# r0 = 0.23
# kernel = a0**2 * george.kernels.Matern32Kernel(r0**2)
# gp = george.GP(kernel, mean = 1.0)

# def lnprob1(p):
#     print p
#     # Trivial improper prior: uniform in the log.
#     if np.any((-10 > p) + (p > 10)):
#         return -np.inf
#     lnprior = 0.0
#     # Update the kernel and compute the lnlikelihood.
#     kernel.pars = np.exp(p)
#     return lnprior + gp.lnlikelihood(y1, quiet=True)

# gp.compute(x1, yerr = e1)
# nwalkers, ndim = 36, len(kernel)
# sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob1)
# p0 = [np.log(kernel.pars) + 1e-4 * np.random.randn(ndim) for i in range(nwalkers)]

# print 'running MCMC on single spectrum'
# p0, _, _ = sampler.run_mcmc(p0, 500)
# samples = np.sqrt(np.exp(sampler.chain[:, 100:, :]))

# ##############
# # plot chain #
# ##############

# fig2 = pl.figure()
# pl.subplot(211)
# for i in range(nwalkers):
#     pl.plot(samples[i,:,0], 'k-', alpha=0.3)
# pl.subplot(212)
# for i in range(nwalkers):
#     pl.plot(samples[i,:,1], 'k-', alpha=0.3)

# ###############
# # corner plot #
# ###############

# samples = samples.reshape(-1,ndim)
# fig3 = corner.corner(samples, labels=['amp','l'], \
#                      show_titles=True, title_args={"fontsize": 12})

#####################
# Rolls Royce model #
#####################

inline = np.zeros(flux.shape, 'bool')
thresh = 0.9
for i in range(nfl):
    inline[i,:] = flux[i,:] < thresh
    l = np.where(flux[i,:] >= thresh)[0]
    np.random.shuffle(l)
    l = l[:len(l)/10]
    inline[i,l] = True
t0 = clock()

a0 = 0.19
r0 = 0.24
kernel = a0**2 * george.kernels.Matern32Kernel(r0**2)
gp = george.GP(kernel, mean = 1.0, solver = george.hodlr.HODLRSolver)

lwav_shift = np.copy(lwav_corr)
wav_shift = np.copy(wav_corr)

ndat = inline.sum()
nseg = 1
n = ndat/nseg
while (ndat/nseg) > nmax:
    nseg += 1
n = ndat/nseg
print nseg, n

def lnprob2(p):
    npar = len(p)
    sigma_prior = 100.0
    lnprior = -0.5 * npar * np.log(2*np.pi) \
      - 0.5 * npar * np.log(sigma_prior) \
      - (p**2/2./sigma_prior**2).sum()
    dlw = p / SPEED_OF_LIGHT
    x = []
    y = []
    e = []
    x.append(wav_corr[0,inline[0,:]])
    y.append(flux[0,inline[0,:]])
    e.append(err[0,inline[0,:]])
    for i in range(npar):
        lws = lwav_corr[i+1,inline[i+1,:]] + dlw[i]
        x.append(np.exp(lws) * 1e10)
        y.append(flux[i+1,inline[i+1,:]])
        e.append(err[i+1,inline[i+1,:]])
    x = np.array([item for sublist in x for item in sublist])
    y = np.array([item for sublist in y for item in sublist])
    e = np.array([item for sublist in e for item in sublist])
    s = np.argsort(x)
    x = x[s]
    y = y[s]
    e = e[s]
    lnlike = 0
    for iseg in range(nseg):
        if iseg == nseg-1:
            xx = x[iseg*n:]
            yy = y[iseg*n:]
            ee = e[iseg*n:]
        else:
            xx = x[iseg*n:(iseg+1)*n]
            yy = y[iseg*n:(iseg+1)*n]
            ee = e[iseg*n:(iseg+1)*n]
        gp.compute(xx, yerr = ee)
        lnlike += gp.lnlikelihood(yy, quiet = True)
    return lnprior + lnlike

nwalkers, ndim = 72, nfl-1
p0 = [np.zeros(ndim) + 10.0 * np.random.randn(ndim) for i in range(nwalkers)]
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob2, threads=8)

print 'running MCMC on velocities: burn in...'
nburn = 50
nstep = 500
p0, _, _ = sampler.run_mcmc(p0, nburn)
print 'production run...'
p0, _, _ = sampler.run_mcmc(p0, nstep)
samples = sampler.chain[:,nburn:,:]

# plot chain 
pl.figure(figsize = (6,ndim))
for i in range(ndim):
    pl.subplot(ndim, 1, i+1)
    for j in range(nwalkers):
        pl.plot(samples[j,:,i], 'k-', alpha=0.3)
pl.savefig('../plots/rollsRoyce_synth_chain.png')

# corner plot
samples = samples.reshape(-1,ndim)
corner.corner(samples, show_titles=True, title_args={"fontsize": 12})
pl.savefig('../plots/rollsRoyce_synth_triangle.png')

# print 16, 50 and 84 percentile values
vals = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), \
           zip(*np.percentile(samples, [16, 50, 84], axis=0)))
p = np.zeros(ndim)
for i in range(ndim):
    print 'dv(%d-0) = %.3f + %.3f -%.3f' % \
      (i+1, vals[i][0],vals[i][1],vals[i][2])
    p[i] = vals[i][0]

t1 = clock()
print 'Time taken %d sec' % (t1-t0)

# final shift and plot:
dlw = p / SPEED_OF_LIGHT
x = []
y = []
e = []
x.append(wav_corr[0,inline[0,:]])
y.append(flux[0,inline[0,:]])
e.append(err[0,inline[0,:]])
for i in range(nfl):
    lws = lwav_corr[i,inline[i,:]] + dlw[i]
    x.append(np.exp(lws) * 1e10)
    y.append(flux[i,inline[i,:]])
    e.append(err[i,inline[i,:]])
x = np.array([item for sublist in x for item in sublist])
y = np.array([item for sublist in y for item in sublist])
e = np.array([item for sublist in e for item in sublist])
s = np.argsort(x)
x = x[s]
y = y[s]
e = e[s]
gp.compute(x, yerr = e)
wav_mu = np.r_[wav.min():wav.max():1001j]
mu, cov = gp.predict(y, wav_mu)
mu_err = np.sqrt(np.diag(cov))
mu2 = gp.predict(y, x, mean_only = True)

pl.figure(figsize=(8,6))
ax1 = pl.subplot(211)
pl.plot(x, y, 'k.')
pl.plot(wav_mu, mu, 'k-')
pl.fill_between(wav_mu, mu + 2 * mu_err, mu - 2 * mu_err, color = 'k', \
                alpha = 0.4)
ax2 = pl.subplot(212,sharex = ax1)
pl.plot(x, y-m2, 'k.')
pl.plot(wav_mu, mu-mu, 'k-')
pl.fill_between(wav_mu, 2 * mu_err, 2 * mu_err, color = 'k', \
                alpha = 0.4)
pl.xlim(wav_mu.min(), wav_mu.max())
pl.savefig('../plots/rollsRoyce_synth_spec.png')

t2 = clock()
print 'Time taken %d sec' % (t2-t1)
