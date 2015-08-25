import numpy as np
import pylab as pl
import glob
import george, emcee
from time import time as clock
import seaborn as sb

###################
# read in spectra #
###################
datadir = '../data/HD127334_HARPSN/'
spectra = np.sort(glob.glob('%s*spec.txt' % datadir))
blaze_files = np.sort(glob.glob('%s*spec.blaze.txt' % datadir))
nfl = len(spectra)
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
######################################################################
# compute photon noise errors & wavelength offset, normalise spectra #
######################################################################
err = np.sqrt(flux)
flux /= b
err /= b
lwav = np.log(wav)
dlw = baryvel / 3.0e8 # need more accurate value of c!
wav_corr = np.copy(wav)
print nfl
for i in np.arange(nfl):
    s = np.argsort(flux[i,:])
    m = flux[i,s[int(0.95*npix)]] # using 95%ile to normalise
    flux[i,:] /= m
    err[i,:] /= m
    wav_corr[i,:] = np.exp(lwav[i,:] + dlw[i])
################################
# training and predictive sets #
################################
x = wav_corr[1,:].flatten()
y = flux[1,:].flatten()
s = err[1,:].flatten()
ss = np.argsort(x)
x = x[ss]
y = y[ss]
s = s[ss]
print 'Full dataset contains %d data points' % len(x)
# l = np.arange(len(x))
# np.random.shuffle(x)
# l = np.sort(l[:4096])
# xt = x[l]
# yt = y[l]
# st = s[l]
xt = np.copy(x)
yt = np.copy(y)
st = np.copy(s)
print 'Training set contains %d data points' % len(xt)
# xp = 10.0**(np.r_[np.log10(x.min()):np.log10(x.max()):4096j])
xp = np.copy(x)
print 'Predictive set contains %d data points' % len(xp)
##########################
# set up GP and train it #
##########################
#k = 0.0225 * george.kernels.ExpSquaredKernel(0.0036)
k = 0.0225 * george.kernels.Matern32Kernel(0.0036)
gp = george.GP(k, mean = 1.0, solver=george.hodlr.HODLRSolver)
print 'training...'
print 'initial HPs:', np.sqrt(gp.kernel.pars)
t0 = clock()
gp.optimize(xt, yt, yerr = st)
t1 = clock()
print 'training took %.1f seconds' % (t1-t0)
print 'best-fit HPs:', np.sqrt(gp.kernel.pars)
##########################################
# now compute GP over full set of inputs #
##########################################
print 'conditioning...'
t0 = clock()
gp.compute(x, yerr = s)
t1 = clock()
print 'conditioning took %.1f seconds' % (t1-t0)
##########################################################
# and finally make prediction for regular grid of points #
##########################################################
print 'predicting...'
t0 = clock()
mu, cov = gp.predict(y, x)
mut, cov = gp.predict(y, xt)
mup, cov = gp.predict(y, xp)
print mu
print mut
print mup
errp = np.sqrt(np.diag(cov))
t1 = clock()
print 'predicting took %.1f seconds' % (t1-t0)
################
# plot results #
################
pl.figure(1)
pl.clf()
ax1 = pl.subplot(211)
pl.plot(x, y, 'k,')
pl.plot(xt, yt, 'k.')
pl.plot(xp, mup, 'r-')
pl.fill_between(xp, mup + 2 * errp, mup - 2 * errp, color = 'r', \
                alpha = 0.4)
pl.ylabel('spectrum')
ax2 = pl.subplot(212, sharex = ax1)
pl.plot(x, y/mu-1, 'k,')
pl.plot(xt, yt/mut-1, 'k,')
pl.plot(xp, mup/mup-1, 'r-')
pl.fill_between(xp, (2 * errp)/mup, (- 2 * errp)/mup, color = 'r', \
                alpha = 0.4)
pl.xlim(x.min(), x.max())
pl.ylabel('residuals')
pl.xlabel('wavelength (Angstroms)')
pl.savefig('HD127334.png')
