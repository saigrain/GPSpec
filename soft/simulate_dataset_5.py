import numpy as np
import pylab as pl
import glob
import GPS_utils as u
from celerite import GP, terms
import scipy.io as sio

SEED = 1234
SPEED_OF_LIGHT = 2.99796458e8 # in m.s
BVMAX = 15e4 # max abs. val of barycentric RV, in m/s
NSMAX = 100 # no. spectra to simulate
SNR = [300,100,30,10]

np.random.seed(SEED)
pl.close('all')

# -------------------------------------
# Read in highest-SNR HD127334 spectrum
# -------------------------------------

# Find all the spectra
datadir = '/Users/aigrain/Soft/GPSpec/data/HD127334_HARPSN/'
spectra = np.sort(glob.glob('%s*spec.txt' % datadir))
blaze_files = np.sort(glob.glob('%s*spec.blaze.txt' % datadir))
K = len(spectra)
# read in spectra & find the highest SNR one
snr_max = 0.0
for i in np.arange(K):
    w, f = np.genfromtxt(spectra[i],skip_header=10).T
    flux = np.copy(f)
    flux_err = np.sqrt(flux/1.4)
    snr = np.median(flux/flux_err)
    print i, snr, snr_max
    if snr > snr_max:
        print 'saving'
        snr_max = snr
        wav = np.copy(w)
        b = np.genfromtxt(blaze_files[i]).T[1]
        # also read in barycentric velocity correction
        f = open(spectra[i])
        r = 0
        while (r < 2):
            w = f.readline().split()
            if w[0] == 'WS_BARY:':
                baryvel = float(w[1])
                r += 1
            if w[0] == 'WS_BJD:':
                bjd = float(w[1])
                r += 1
        f.close()
# blaze-correct spectrum
flux /= b
flux_err /= b
# evaluate rest wavelength
print wav
lwav = np.log(wav * 1e-10) # input wavelengths are in Angstrom
dlw = baryvel / 2.99796458e8 # in m
wav_rest = np.exp(lwav + dlw) * 1e9 # my code wants wavelengths in nm
print wav_rest

# ---------------------------------
# Model observed spectrum with a GP
# ---------------------------------
lwav = np.log(wav_rest * 1e-9) # in m
lw0, lw1 = lwav.min(), lwav.max()
x = (lwav-lw0) / (lw1-lw0)
s = np.argsort(flux)
N = len(flux)
m = flux[s[int(0.98*N)]]
y = flux / m
yerr = flux_err / m
HP_in = np.array([-1.65, -5.49])
xpred = np.linspace(0.29,0.34,1000)
HPs, mu, std = u.Fit0(x, y, yerr, verbose = True, doPlot = False, \
             xpred = xpred, HP_init = HP_in)
pl.clf()
pl.errorbar(x,y,yerr=yerr,fmt='k.',ms=8,capsize=0,lw=0.5,alpha=0.5)
pl.fill_between(xpred, mu + 2 * std, mu - 2 * std, alpha=0.2, color='C0', lw=0)
pl.fill_between(xpred, mu + std, mu - std, alpha=0.2, color='C0', lw=0)
pl.plot(xpred, mu, 'C0-')
pl.xlim(xpred.min(), xpred.max())
pl.ylim((mu - 5 * std).min(), (mu + 5 * std).max()) 

# ----------------
# Simulate spectra
# ----------------
# First set up GP object
k = terms.Matern32Term(log_sigma = HPs[0], log_rho = HPs[1])
gp = GP(k, mean = 1.0)
gp.compute(x, yerr = yerr)
# Barycentric shifts
baryvel = np.random.uniform(low = -BVMAX, high = BVMAX, size = NSMAX)
dlw = baryvel / SPEED_OF_LIGHT
lwav_sim = np.tile(lwav, (NSMAX, 1)) + dlw[:,None]
x_sim = (lwav_sim - lw0) / (lw1 - lw0)
wav_rest_sim = np.exp(lwav_sim) * 1e9
wav_earth_sim = np.tile(wav_rest, (NSMAX, 1))
# Evaluate each spectrum using predictive mean of GP conditioned on
# observed spectrum, and wavelength shifts caused by barycentric
# velocity changes
flux_sim = np.zeros((NSMAX, N))
for i in range(NSMAX):
        print baryvel[i], dlw[i], np.median(x_sim[i,:]-x)*N
        print 'Simulating spectrum %d' % (i + 1)
        xpred = x_sim[i,:].flatten()
        mu, _ = gp.predict(y, xpred, return_var = True)
        flux_sim[i,:] = mu
pl.figure(figsize = (8,6))
ax1 = pl.subplot(411)
pl.plot(wav_rest_sim[:3,:].T, flux_sim[:3,:].T, '.', ms = 8, alpha=0.5,lw=0.5)
pl.ylabel('rest')
ax2 = pl.subplot(412, sharex = ax1, sharey = ax1)
pl.plot(wav_earth_sim[:3,:].T, flux_sim[:3,:].T, '-', alpha=0.5,lw=0.5)
pl.ylabel('true')
# Add noise
for j,snr in enumerate(SNR):
    print j, snr
    sigma = (flux_sim/snr) 
    print np.median(sigma)
    flux_noisy = np.copy(flux_sim)
    for i in np.arange(NSMAX):
        noise = np.random.normal(0,1,N) * sigma[i,:]
        flux_noisy[i,:] += noise
    if j == 0:
        ax3 = pl.subplot(4, 1, 3, sharex = ax1, sharey = ax1)
        pl.plot(wav_earth_sim[:3,:].T, flux_noisy[:3,:].T, '-', lw=0.5, alpha=0.5)
        pl.ylabel('SNR %d' % snr)
    if j == (len(SNR)-1):
        ax4 = pl.subplot(4, 1, 4, sharex = ax1, sharey = ax1)
        pl.plot(wav_earth_sim[:3,:].T, flux_noisy[:3,:].T, '-', lw=0.5, alpha=0.5)
        pl.ylabel('SNR %d' % snr)
    sio.savemat('../data/synth_dataset_005_%04d.mat' % snr, \
                    {'wavelength': wav_earth_sim, 'flux': flux_noisy, \
                         'error': sigma, 'baryvel': baryvel})
pl.xlim(wav_earth_sim.min(), wav_earth_sim.max())
pl.ylim(0,1.3)
pl.xlabel('wavelength (nm')
pl.tight_layout()
pl.draw()
pl.savefig('../plots/synth_dataset_005.png')
