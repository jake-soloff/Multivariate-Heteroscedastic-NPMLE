##### APOGEE analysis (Figure 6) #####

## public github repository: 
## https://github.com/jake-soloff/ebpy
from ebpy.GLMixture import GLMixture

import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt

## Load observations
X = pkl.load(open('APOGEE/X.pkl', 'rb'))

## Load precision matrices (diagonal)
prec = pkl.load(open('APOGEE/prec.pkl', 'rb'))

## Construct legend for columns of APOGEE data
abunds = ['MG_FE', 'FE_H', 'C_FE', 'CI_FE', 'N_FE', 'O_FE', 
          'AL_FE', 'CA_FE', 'TI_FE', 'SI_FE', 'P_FE', 'S_FE', 
          'K_FE', 'TIII_FE', 'V_FE', 'CR_FE', 'MN_FE', 'CO_FE', 'NI_FE']
n, d = X.shape
inds = [i for i in range(d) if abunds[i] in ['MG_FE','SI_FE']]

## Restrict to two-dimensional dataset MG/FE and SI/FE
X = X[:, inds]
prec = prec[:, inds]
abunds = [abunds[i] for i in inds]

## Define grid of atoms (note all diagonal covariance matrices)
a,b = np.meshgrid(np.linspace(np.min(X[:,0]), np.max(X[:,0]), 100), 
                  np.linspace(np.min(X[:,1]), np.max(X[:,1]), 100))
atoms = np.vstack([np.ravel(a), np.ravel(b)]).T

## Fit NPMLE using ebpy
m = GLMixture(prec_type='diagonal')
m.atoms_init = atoms
m.fit(X, prec, max_iter_em=0, n_chunks=20, log_prob_thresh=-100)
atoms, weights = m.get_params()

## Compute posterior means
gmleb = m.posterior_mean(X, prec)

## Plot raw data, EB posterior means, initial grid and final estimator
plt.rcParams.update({'font.size': 16})

fig, ax = plt.subplots(2,2, sharex=True, sharey=True, figsize=(14,14))

ax[0,0].scatter(X[:,1], X[:,0], s=1, alpha = 0.25, color='black')
ax[0,1].scatter(gmleb[:,1], gmleb[:, 0], s=1, alpha = 0.25, color='blue')

ax[0,0].text(-.375, .48, r'Raw Data', fontsize=18) # Observed Values
ax[0,1].text(-.375, .48, r'Empirical Bayes', fontsize=18) # GMLEB Estimates

ax[0,0].set_xlim(atoms[:,1].min(),atoms[:,1].max())
ax[0,0].set_ylim(atoms[:,0].min(),atoms[:,0].max())
ax[1,0].scatter(atoms_[:,1], atoms_[:,0], marker='o', color='r', s=.5, alpha = 0.8)

rgbs = np.zeros((len(weights),4))
rgbs[:, 0] = 1.0
rgbs[:, 3] = (np.maximum(weights.flatten(), 0) / np.max(weights))
ax[1,1].scatter(atoms[:,1], (atoms[:,0]), marker='o', color='r', s=50*rgbs[:, 3])

ax[1,0].text(-.375, .48, r'Initial Grid', fontsize=18)
ax[1,1].text(-.375, .48, r'Final Estimator $\widehat{G}_n$', fontsize=18)

ax[1,0].set_xlabel('[Mg/Fe]')
ax[1,1].set_xlabel('[Mg/Fe]')
ax[0,0].set_ylabel('[Si/Fe]')
ax[1,0].set_ylabel('[Si/Fe]')

plt.tight_layout()

plt.savefig('apogee.png', dpi=500)
