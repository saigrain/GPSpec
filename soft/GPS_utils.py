import numpy as np
import george



def GPFit(x, y, yerr = None, kernel = 'SE', par_init = [1.0, 0.01], \
          verbose = True):
    if kernel == 'SE':
        k = par_init[0] * george.kernels.ExpSquaredKernel(par_init[1])
    gp = george.GP(k, mean = 1.0)
    gp.compute(x, yerr = yerr)
    gp.optimize(x, y, yerr=yerr)
    if verbose:
        print 'Initial pars:', par_init
        print 'Fitted pars:', gp.kernel.pars
    mu, cov = gp.predict(y, x)
    err = np.sqrt(np.diag(cov))
    return mu, err, gp.kernel.pars
