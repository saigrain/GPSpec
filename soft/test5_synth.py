import numpy as np
import pylab as pl
import glob
import george, emcee, corner
from time import time as clock
import scipy.io as sio
import scipy.optimize as op

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

t0 = clock()

a0 = 0.19
r0 = 0.24
kernel = a0**2 * george.kernels.Matern32Kernel(r0**2)
gp = george.GP(kernel, mean = 1.0, solver = george.hodlr.HODLRSolver)

lwav_shift = np.copy(lwav_corr)
wav_shift = np.copy(wav_corr)

ndat = npix*nfl
nseg = 1
n = ndat/nseg
while (ndat/nseg) > nmax:
    nseg += 1
n = ndat/nseg
print nseg, n

def nll(p):
    npar = len(p)
    pp = np.append(p, -p.sum())
    dlw = pp / SPEED_OF_LIGHT
    for i in range(npar+1):
        lwav_shift[i] = lwav_corr[i,:] + dlw[i]
        wav_shift[i] = np.exp(lwav_shift[i]) * 1e10
    x = wav_shift.flatten()
    y = flux.flatten()
    e = err.flatten()
    s = np.argsort(x)
    x = x[s]
    y = y[s]
    e = e[s]
    pl.clf()
    pl.plot(x, y, 'k.')
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
        pl.plot(xx, yy, 'r.')
        gp.compute(xx, yerr = ee)
        mu = gp.predict(yy, xx, mean_only = True)
        pl.plot(xx, mu, 'b-')
        ll = gp.lnlikelihood(yy, quiet = True)
        lnlike += ll
        print iseg, ll, lnlike
        pl.show()
        raw_input()
    print p[:5], -lnlike
    raw_input()
    return -lnlike

print 'fitting for velocities'
ndim = nfl-1
p0 = np.zeros(ndim)
p1 = op.fmin_bfgs(nll,p0,disp=True)
p = np.append(p1, p1.sum())
for i in range(ndim+1):
    print 'delta v(%d-0) = %.3f' % (i+1, p[i])

t1 = clock()
print 'Time taken %d sec' % (t1-t0)

