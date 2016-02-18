import numpy as np
import pylab as pl
import glob
import george, emcee, corner
from time import time as clock
import scipy.io as sio

SPEED_OF_LIGHT = 2.99796458e8
nmax = 4096

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
# ifl = np.array([0,3,7,10,14,17,21,24,28,32])
ifl = np.arange(35).astype(int)
nfl = len(ifl)
wav = wav[ifl,:]
flux = flux[ifl,:]
err = err[ifl,:]
# ipix = np.r_[1000:2000].astype(int)
# npix = len(ipix)
# wav = wav[:,ipix]
# flux = flux[:,ipix]
# err = err[:,ipix]
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
    print inline[i,:].sum()

t0 = clock()

a0 = 0.19
r0 = 0.24
kernel = a0**2 * george.kernels.Matern32Kernel(r0**2)
gp = george.GP(kernel, mean = 1.0, solver = george.hodlr.HODLRSolver)
print kernel.pars
print np.sqrt(kernel.pars)

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
    pp = np.append(p, -p.sum())
    dlw = pp / SPEED_OF_LIGHT
    for i in range(npar+1):
        lwav_shift[i] = lwav_corr[i,:] + dlw[i]
        wav_shift[i] = np.exp(lwav_shift[i]) * 1e10
    x = wav_shift[inline].flatten()
    y = flux[inline].flatten()
    e = err[inline].flatten()
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
    
nwalkers, ndim = 100, nfl-1
p0 = [np.zeros(ndim) + 10.0 * np.random.randn(ndim) for i in range(nwalkers)]
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob2, threads=8)

print 'running MCMC on velocities: burn in...'
nburn = 100
nstep = 500
p0, _, _ = sampler.run_mcmc(p0, nburn)
print 'production run...'
p0, _, _ = sampler.run_mcmc(p0, nstep)
samp = sampler.chain
samples = np.zeros((nwalkers,nstep,ndim+1))
samples[:,:,:ndim] = samp[:,nburn:,:]
for i in range(nwalkers):
    for j in range(nstep):
        samples[i,j,-1] = samp[i,nburn+j,:].sum()

# plot chain 
pl.figure(figsize = (6,(ndim+1)/2.))
for i in range(ndim+1):
    pl.subplot(ndim+1, 1, i+1)
    for j in range(nwalkers):
        pl.plot(samples[j,:,i], 'k-', alpha=0.3)
pl.savefig('../plots/rollsRoyce_synth_chain.png')

# corner plot
samples = samples.reshape(-1,ndim+1)
corner.corner(samples, show_titles=True, title_args={"fontsize": 12})
pl.savefig('../plots/rollsRoyce_synth_triangle.png')

# print 16, 50 and 84 percentile values
vals = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), \
           zip(*np.percentile(samples, [16, 50, 84], axis=0)))
p = np.zeros(ndim+1)
for i in range(ndim+1):
    print 'delta v(%d-0) = %.3f + %.3f -%.3f' % \
      (i+1, vals[i][0],vals[i][1],vals[i][2])
    p[i] = vals[i][0]

t1 = clock()
print 'Time taken %d sec' % (t1-t0)

# final shift and plot:
dlw = p / SPEED_OF_LIGHT
for i in range(len(p)):
    lwav_shift[i,:] = lwav_corr[i,:] + dlw[i]
    wav_shift[i,:] = np.exp(lwav_shift[i]) * 1e10
x = wav_shift.flatten()
y = flux.flatten()
e = err.flatten()
s = np.argsort(x)
x = x[s]
y = y[s]
e = e[s]

ndat = inline.sum()
nseg = 1
n = ndat/nseg
while (ndat/nseg) > nmax:
    nseg += 1
n = ndat/nseg
print nseg, n

mu = np.zeros(len(x)) + np.nan
mu_err = np.zeros(len(x)) + np.nan
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
    gp.compute(xx, yerr = ee, sort = True)
    m, c = gp.predict(yy, xx)
    me = np.sqrt(np.diag(c))
    if iseg == nseg-1:
        mu[iseg*n:] = m
        mu_err[iseg*n:] = me
    else:
        mu[iseg*n:(iseg+1)*n] = m
        mu_err[iseg*n:(iseg+1)*n] = me

pl.figure()
ax1=pl.subplot(211)
pl.plot(wav_shift.T, flux.T, '.')
pl.plot(x, mu, 'k-')
pl.fill_between(x, mu + 2 * mu_err, mu - 2 * mu_err, color = 'k', \
                alpha = 0.4)
pl.axhline(thresh, ls = '--', color = 'k')
pl.subplot(212,sharex=ax1)
pl.plot(x, y-mu, 'k.')
pl.plot(x, mu-mu, 'k-')
pl.fill_between(x, 2 * mu_err, - 2 * mu_err, color = 'k', \
                alpha = 0.4)
pl.savefig('../plots/rollsRoyce_synth_spec.png')

t1 = clock()
print 'Time taken %d sec' % (t1-t0)
