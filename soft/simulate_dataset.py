import numpy as np
import pylab as pl
import scipy.io as sio
from astropy.convolution import convolve, Gaussian1DKernel

SPEED_OF_LIGHT = 2.99796458e8
nsim = 5
wmin = 5000 # Angstrom
wmax = 6000
npix = 10000
resol = 10 # 10 pix = 1 Angstrom

##############################
# Simulate synthetic dataset #
##############################

dwav = (wmax - wmin) / float(npix)
wav = np.arange(npix) * dwav + wmin

def gauss1(x,a,mu,sig):
    y = a * np.exp(-(x-mu)**2/2/sig**2)
    return y

# Define properties of spectral lines
nlines = 100
np.random.seed(2)
m = np.random.uniform(wmin, wmax,nlines)
s = 10.0**np.random.uniform(-1,0.5,nlines)
# even the narrowest lines are well sampled
a = 0.05*10.0**np.random.uniform(0,1,nlines)

# Define "barycentric" velocity shifts (chose those so as to span most
# of a pixel)
baryvel_max = SPEED_OF_LIGHT * 2 / (wmax + wmin)
baryvel = np.random.uniform(high = baryvel_max, size = nsim)
print baryvel
# Define additional intrinsic velocity shifts of up to 200 km/s
starvel = np.append(0, np.sort(np.random.uniform(high = 200.0e3, size=nsim-1)))
print starvel

lwav = np.log(wav * 1e-10)
dlw = (baryvel + starvel) / SPEED_OF_LIGHT
print dlw
lwav_rest = np.zeros((nsim, npix))
wav_rest = np.zeros((nsim, npix))
for i in np.arange(nsim):
    lwav_rest[i,:] = lwav + dlw[i]
wav_rest = np.exp(lwav_rest) * 1e10
wav_earth = np.tile(wav, (nsim, 1))

# Simulate emitted spectra
flux = np.ones((nsim, npix))
for i in np.arange(nsim):
    for j in np.arange(nlines):
        flux[i,:] -= gauss1(wav_rest[i,:], a[j], m[j], s[j])
pl.close('all')
pl.figure(figsize = (8,6))
ax1 = pl.subplot(311)
pl.plot(wav_rest.T, flux.T, 'k,')
pl.ylabel('true')

# Degrade resolution
flux_deg = np.copy(flux)
for i in np.arange(nsim):
    flux_deg[i,:] = convolve(flux[i,:], Gaussian1DKernel(2 * resol))
# get rid of the edges which are screwed up by the convolution
wav_deg = wav_earth[:,40:-40]
flux_deg = flux_deg[:,40:-40]
# Resample to Nyquist sampling
for i in np.arange(nsim):
    if i == 0:
        tmp = wav_deg[i,::resol]
        npix_new = len(tmp)
        wav_obs = np.zeros((nsim,npix_new))
        flux_obs = np.zeros((nsim,npix_new))
        wav_obs[i,:] = tmp
    else:
        wav_obs[i,:] = wav_deg[i,::resol]
    flux_obs[i,:] = flux_deg[i,::resol]
ax2 = pl.subplot(312, sharex = ax1, sharey = ax1)
pl.plot(wav_deg.T, flux_deg.T, 'k-')
pl.plot(wav_obs.T, flux_obs.T, '.')
pl.ylabel('degraded')

# Add noise
sigma = 0.01 * np.sqrt(flux_obs)
flux_noisy = np.copy(flux_obs)
for i in np.arange(nsim):
    noise = np.random.normal(0,1,npix_new) * sigma[i,:]
    flux_noisy[i,:] += noise
ax3 = pl.subplot(313, sharex = ax1, sharey = ax1)
pl.plot(wav_obs.T, flux_noisy.T, '.')
pl.ylabel('noisy')
pl.xlim(wmin, wmax)
pl.xlabel('wavelength (Angstrom')
pl.savefig('../plots/synth_dataset_003.png')
    
sio.savemat('../data/synth_dataset_003.mat', \
            {'wavelength': wav_obs, \
             'flux': flux_noisy, \
             'error': sigma, \
             'baryvel': baryvel, \
             'starvel': starvel})
