import numpy as np
import pylab as pl
import glob
import george
from time import time as clock

def asymclip(a, itermax = 10, nsig = 3):
    m = np.median(a)
    r = a - m
    ll = r > 0
    s = 1.48 * np.median(abs(r[ll]))
    l = r > - nsig * s
    return l

###################
# read in spectra #
###################
datadir = '../data/HD127334_HARPSN/'
spectra = np.sort(glob.glob('%s*spec.txt' % datadir))
blaze_files = np.sort(glob.glob('%s*spec.blaze.txt' % datadir))
nfl = 1
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
# work on 1st obs only for now #
################################
x = wav_corr.flatten()
y = flux.flatten()
s = err.flatten()
print 'Dataset contains %d data points' % len(x)
##################################################
# select "continuum" using simple sigma clipping #
##################################################
#l = asymclip(y)
#xt = x[l]
#yt = y[l]
#st = s[l]
##########################
# set up GP and train it #
##########################
k = 0.001 * george.kernels.Matern32Kernel(0.001)
gp = george.GP(k, mean = 1.0, solver=george.hodlr.HODLRSolver)
print np.sqrt(gp.kernel.pars)
# gp.compute(x, yerr = s)
gp.optimize(x, y, yerr = s, dims=0)
print np.sqrt(gp.kernel.pars)
mu, cov = gp.predict(y, x)
err = np.sqrt(np.diag(cov))
pl.clf()
ax1=pl.subplot(211)
pl.plot(x, y, 'k.')
pl.plot(x, mu, 'r-')
pl.fill_between(x, mu + 2 * err, mu - 2 * err, color = 'r', \
                alpha = 0.4)
pl.subplot(212,sharex=ax1)
pl.plot(x, y-mu, 'k.')
pl.plot(x, mu-mu, 'r-')
pl.fill_between(x, 2 * err, - 2 * err, color = 'r', \
                alpha = 0.4)
