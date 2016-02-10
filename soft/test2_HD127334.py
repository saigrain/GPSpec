import numpy as np
import pylab as pl
import glob
import george, emcee, corner

SPEED_OF_LIGHT = 2.99796458e8

###################
# read in spectra #
###################
datadir = '../data/HD127334_HARPSN/'
spectra = np.sort(glob.glob('%s*spec.txt' % datadir))
blaze_files = np.sort(glob.glob('%s*spec.blaze.txt' % datadir))
nfl = len(spectra)
print 'Reading in %d spectra' % nfl
baryvel = np.zeros(nfl)
bjd = np.zeros(nfl)
for i in np.arange(nfl):
    w,f=np.genfromtxt(spectra[i],skip_header=10).T
    b = np.genfromtxt(blaze_files[i]).T[1]
    if i == 0:
        npix = len(f)
        wav = np.zeros((nfl,npix))
        flux = np.zeros((nfl, npix))
        blaze = np.zeros((nfl, npix))
    wav[i,:] = w
    flux[i,:] = f
    blaze[i,:] = b
    f = open(spectra[i])
    r = 0
    # also read in barycentric velocity correction
    while (r < 2):
        w = f.readline().split()
        if w[0] == 'WS_BARY:':
            baryvel[i] = float(w[1])
            r += 1
        if w[0] == 'WS_BJD:':
            bjd[i] = float(w[1])
            r += 1
    f.close()

###############################
# work on subset only for now #
###############################
print 'Selecting subset'
ifl = np.array([0,7,14,21,28])
nfl = len(ifl)
wav = wav[ifl,:]
flux = flux[ifl,:]
blaze = blaze[ifl,:]
baryvel = baryvel[ifl]
bjd = bjd[ifl]
ipix = np.r_[1000:2000].astype(int)
npix = len(ipix)
wav = wav[:,ipix]
flux = flux[:,ipix]
blaze = blaze[:,ipix]

###################################################
# compute photon noise errors & normalise spectra #
###################################################
print 'Normalising and computing errors'
err = np.sqrt(flux)
flux /= blaze
err /= blaze
lwav = np.log(wav*1e-10)
dlw = baryvel / SPEED_OF_LIGHT
lwav_corr = np.copy(lwav)
wav_corr = np.copy(wav)
for i in np.arange(nfl):
    s = np.argsort(flux[i,:])
    m = flux[i,s[int(0.95*npix)]] # using 95%ile to normalise
    flux[i,:] /= m
    err[i,:] /= m
    lwav_corr[i,:] += dlw[i]
    wav_corr[i,:] = np.exp(lwav_corr[i,:]) * 1e10

################
# plot spectra #
################
# pl.figure()
# pl.plot(wav_corr.T, flux.T)

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

# wav_corr = wav_corr[:,250:750]
# lwav_corr = lwav_corr[:,250:750]
# flux = flux[:,250:750]
# err = err[:,250:750]

a0 = 0.19
r0 = 0.24
kernel = a0**2 * george.kernels.Matern32Kernel(r0**2)
gp = george.GP(kernel, mean = 1.0, solver = george.hodlr.HODLRSolver)
print kernel.pars
print np.sqrt(kernel.pars)

lwav_shift = np.copy(lwav_corr)
wav_shift = np.copy(wav_corr)

def lnprob2(p):
    print p
    npar = len(p)
    sigma_prior = 100.0
    lnprior = -0.5 * npar * np.log(2*np.pi) \
      - 0.5 * npar * np.log(sigma_prior) \
      - (p**2/2./sigma_prior**2).sum()
    dlw = p / SPEED_OF_LIGHT
    for i in range(npar):
        lwav_shift[i+1,:] = lwav_corr[i+1,:] + dlw[i]
        wav_shift[i+1,:] = np.exp(lwav_shift[i+1]) * 1e10
    x = wav_shift.flatten()
    y = flux.flatten()
    e = err.flatten()
    gp.compute(x, yerr = e, sort = True)
    lnlike = gp.lnlikelihood(y, quiet = True)
    print lnprior, lnlike
    return lnprior + lnlike
    
nwalkers, ndim = 36, nfl-1
p0 = [np.zeros(ndim) + np.random.randn(ndim) for i in range(nwalkers)]
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob2)

print 'running MCMC on velocities'
p0, _, _ = sampler.run_mcmc(p0, 100)
p0, _, _ = sampler.run_mcmc(p0, 500)
samples = sampler.chain

# plot chain 
pl.figure()
for i in range(ndim):
    pl.subplot(ndim, 1, i+1)
    for j in range(nwalkers):
        pl.plot(samples[j,:,i], 'k-', alpha=0.3)

# corner plot
samples = samples.reshape(-1,ndim)
corner.corner(samples, show_titles=True, title_args={"fontsize": 12})

# print 16, 50 and 84 percentile values
vals = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), \
           zip(*np.percentile(samples, [16, 50, 84], axis=0)))
p = np.zeros(ndim)
for i in range(ndim):
    print 'delta v(%d-0) = %.3f + %.3f -%.3f' % \
      (i+1, vals[i][0],vals[i][1],vals[i][2])
    p[i] = vals[i][0]

# final shift and plot:
dlw = p / SPEED_OF_LIGHT
for i in range(len(p)):
    lwav_shift[i+1,:] = lwav_corr[i+1,:] + dlw[i]
    wav_shift[i+1,:] = np.exp(lwav_shift[i+1]) * 1e10
x = wav_shift.flatten()
y = flux.flatten()
e = err.flatten()
s = np.argsort(x)
x = x[s]
y = y[s]
e = e[s]
gp.compute(x, yerr = e, sort = True)
mu, cov = gp.predict(y, x)
mu_err = np.sqrt(np.diag(cov))

pl.figure()
ax1=pl.subplot(211)
pl.plot(wav_shift.T, flux.T, '.')
pl.plot(x, mu, 'k-')
pl.fill_between(x, mu + 2 * mu_err, mu - 2 * mu_err, color = 'k', \
                alpha = 0.4)
pl.subplot(212,sharex=ax1)
pl.plot(x, y-mu, 'k.')
pl.plot(x, mu-mu, 'k-')
pl.fill_between(x, 2 * mu_err, - 2 * mu_err, color = 'k', \
                alpha = 0.4)
