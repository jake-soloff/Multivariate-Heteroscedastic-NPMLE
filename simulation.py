##### Simulation (Figure 2) #####

## public github repository: 
## https://github.com/jake-soloff/ebpy
from ebpy.GLMixture import GLMixture

import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt

## Generate observations and covariance matrices
np.random.seed(1234567891)

n,p = 1000,2
ang = 2*np.pi*np.random.rand(n)
truth = 2*np.array([np.cos(ang), np.sin(ang)]).T

l, u = .5, 1
covs = l + np.random.rand(n,p)*(u-l)
prec = 1/covs
X = truth + covs**(1/2)*np.random.randn(n,p)

## Define a Gaussian location mixture using ebpy
m = GLMixture(prec_type='diagonal') 
L = np.max(np.abs(X))
(XX, YY) = np.meshgrid(np.linspace(-L, L, 50), np.linspace(-L, L, 50))
atoms = np.vstack([XX.flatten(), YY.flatten()]).T

## Compute the NPMLE using ebpy
m.atoms_init = atoms
m.fit(X, prec, max_iter_em=1000)

## Denoised estimates based on empirical prior
gmleb = m.posterior_mean(X, prec)

## Define the oracle
o = GLMixture(prec_type='diagonal')
o.set_params(atoms=truth, weights=np.ones(n)/n)

## Denoised estimates based on oracle prior
obayes = o.posterior_mean(X, prec)

plt.rcParams.update({'font.size': 22,
                     'mathtext.fontset': 'stix',
                     'font.family': 'STIXGeneral'})
fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(12,12))

ax[0,0].plot(X[:,0], X[:,1], 'k.')
ax[0,1].plot(truth[np.argsort(ang),0], truth[np.argsort(ang),1], 'k-', linewidth=3)
ax[0,1].scatter(m.atoms[:, 0], m.atoms[:, 1], s=m.weights*500, c='r')

ax[1,0].plot(obayes[:,0], obayes[:,1], 'k.')
ax[1,1].plot(gmleb[:,0], gmleb[:,1], 'k.')

M = 4
plt.xlim([-M, M])
plt.ylim([-M, M])

ax[0,1].text(-3.75, 3.4, 'Prior $G^*$, and')
ax[0,1].text(-.95, 3.4, 'NPMLE $\\widehat{G}_n$', c='r')
ax[0,0].text(-3.75, 3.4, 'Raw Data $Y_i$')
ax[1,0].text(-3.75, 3.4, 'Oracle Bayes $\\hat\\theta^*_i$')
ax[1,1].text(-3.75, 3.4, 'Empirical Bayes $\\hat\\theta_i$')

plt.tight_layout()
plt.savefig('circle_demo.png', dpi=200)
