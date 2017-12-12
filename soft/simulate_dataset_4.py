import numpy as np
import pylab as pl
import scipy.io as sio
from astropy.convolution import convolve, Gaussian1DKernel

SEED = 1234
SPEED_OF_LIGHT = 2.99796458e8 # in m.s
WCEN = 550.0 # in nm
R = 1.0e5 # resolution of final spectra
DWPIX = WCEN / (2*R) # pixel scale of final spectra (in nm)
NPIXMAX = 4096 # no. pixels of final spectra
WMIN = WCEN - NPIXMAX * DWPIX / 2
WMAX = WCEN + NPIXMAX * DWPIX / 2
NLINES = 200
BVMAX = 3.0e4 # max abs. val of barycentric RV, in m/s
NSMAX = 128 # no. spectra to simulate
SNR = [300,30,10]

np.random.seed(SEED)

# Define super-resolved wavelength arrays in star and barycentric frame
baryvel = np.random.uniform(low = -BVMAX, high = BVMAX, size = NSMAX)
wav = np.arange(NPIXMAX) * DWPIX + WMIN
lwav = np.log(wav * 1e-9)
dlw = baryvel / SPEED_OF_LIGHT
lwav_rest = np.tile(lwav, (NSMAX, 1)) + dlw[:,None]
wav_rest = np.exp(lwav_rest) * 1e9
wav_earth = np.tile(wav, (NSMAX, 1))

# Define properties of spectral lines. Will be used in all spectra
def gauss1(x,a,mu,sig):
    y = a * np.exp(-(x-mu)**2/2/sig**2)
    return y
m = np.random.uniform(WMIN, WMAX, NLINES)
s = 10.0**np.random.uniform(-2,-1.8,NLINES)
# even the narrowest lines are well sampled
a = 0.5*10.0**np.random.uniform(-1,0,NLINES)

# Simulate spectra
flux = np.ones((NSMAX, NPIXMAX))
for i in np.arange(NSMAX):
    for j in np.arange(NLINES):
        flux[i,:] -= gauss1(wav_rest[i,:], a[j], m[j], s[j])
pl.close('all')
pl.figure(figsize = (8,6))
ax1 = pl.subplot(411)
pl.plot(wav_earth[:3,:].T, flux[:3,:].T, '-', alpha=0.5,lw=0.5)
pl.ylabel('true')

# Add noise
for j,snr in enumerate(SNR):
    sigma = (1./snr) * np.sqrt(flux)
    print np.median(sigma)
    flux_noisy = np.copy(flux)
    for i in np.arange(NSMAX):
        noise = np.random.normal(0,1,NPIXMAX) * sigma[i,:]
        flux_noisy[i,:] += noise
    axc = pl.subplot(4, 1, j+2, sharex = ax1, sharey = ax1)
    pl.plot(wav_earth[:3,:].T, flux_noisy[:3,:].T, '-', lw=0.5, alpha=0.5)
    pl.ylabel('SNR %d' % snr)
    sio.savemat('../data/synth_dataset_004_%04d.mat' % snr, \
                    {'wavelength': wav_earth, 'flux': flux_noisy, \
                         'error': sigma, 'baryvel': baryvel})

pl.xlim(WMIN, WMAX)
pl.ylim(0,1.3)
pl.xlabel('wavelength (nm')
pl.tight_layout()
pl.draw()
pl.savefig('../plots/synth_dataset_004.png')
