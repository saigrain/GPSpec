import numpy as np
import matplotlib.pyplot as pl
from astropy.convolution import convolve, Gaussian1DKernel
import george

def gauss1(x,a,mu,sig):
    y = (a / np.sqrt(2 * np.pi) / sig) * np.exp(-(x-mu)**2/2/sig**2)
    return y

x_true = np.arange(0,1000,0.1) 
y_true = np.ones(len(x_true))
nlines = 100
np.random.seed(1)
m = np.random.uniform(x_true.min(),x_true.max(),nlines)
s = 10.0**np.random.uniform(np.log10(0.3),1,nlines) # even the narrowest lines are well
a = 0.05*10.0**np.random.uniform(0,1,nlines)
for i in np.arange(len(m)):
    y_true -= gauss1(x_true, a[i], m[i], s[i])
print '"True" spectrum'
pl.plot(x_true,y_true)
pl.draw()
raw_input('continue?')

y_degraded = convolve(y_true, Gaussian1DKernel(20)) # 10 pix = 1 wav unit
# get rid of the edges which are screwed up by the convolution
x_degraded = x_true[20:-20]
y_degraded = y_degraded[20:-20]
print 'degraded spectrum'
pl.plot(x_degraded, y_degraded)
raw_input('continue?')

nsim = 5
dx = 10 
npix = int(np.floor(len(x_degraded)/dx))
x_obs = np.zeros((nsim,npix))
y_obs = np.zeros((nsim,npix))
s_obs = np.zeros((nsim,npix))
istart = np.random.uniform(0,dx,nsim).astype(int)
for i in np.arange(nsim):
    x_obs[i,:] = x_degraded[istart[i]:istart[i]+npix*dx:dx]
    y_obs[i,:] = y_degraded[istart[i]:istart[i]+npix*dx:dx]
    sigma = 0.01 * np.sqrt(y_obs[i,:])
    s_obs[i,:] = sigma
    noise = np.random.normal(0,1,npix) * sigma
    y_obs[i,:] += noise
print 'observed spectra'
pl.plot(x_obs.T,y_obs.T,'.');
pl.draw()
raw_input('continue?')

k = 1.0 * george.kernels.ExpSquaredKernel(1.0)
gp = george.GP(k)
x = x_obs[4,:].flatten()
y = y_obs[4,:].flatten()
s = s_obs[4,:].flatten()
ss = np.argsort(x)
x = x[ss]
y = y[ss]
s = s[ss]
gp.compute(x, yerr = s)
gp.optimize(x, y, yerr=s)
print np.sqrt(gp.kernel.pars)
gp.compute(x, yerr = s)
mu, cov = gp.predict(y, x_true)
err = np.sqrt(np.diag(cov))
print 'conditioned on one observation only'
pl.plot(x_true, mu, 'r-')
pl.fill_between(x_true, mu + 2 * err, mu - 2 * err, color = 'r', alpha = 0.2)
pl.draw()
raw_input('continue?')

k = 1.0 * george.kernels.ExpSquaredKernel(0.01)
gp2 = george.GP(k)
x = x_obs.flatten()
y = y_obs.flatten()
s = s_obs.flatten()
ss = np.argsort(x)
x = x[ss]
y = y[ss]
s = s[ss]
gp2.compute(x, yerr = s)
gp2.optimize(x, y, yerr=s)
print np.sqrt(gp2.kernel.pars)
gp2.compute(x, yerr = s)
mu2, cov = gp2.predict(y, x_true)
err2 = np.sqrt(np.diag(cov))
print 'conditioned on one 5 observations'
pl.plot(x_true, mu2, 'g-')
pl.fill_between(x_true, mu2 + 2 * err2, mu2 - 2 * err2, color = 'g', \
                alpha = 0.2)
pl.draw()
raw_input('continue?')
