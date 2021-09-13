##### Analyze Gaia-TGAS data (Figures 1 & 5) #####

## public github repository: 
## https://github.com/jake-soloff/ebpy
from ebpy.GLMixture import GLMixture

import pickle as pkl
import numpy as np
import sys

## Load data
df = pkl.load(open('CMD_obs.pkl', 'rb'))

n = len(df)
s = int(sys.argv[1])

X = np.array(df[['color','absMagKinda']])
prec = 1/np.array(df[['color_err', 'absMagKinda_err']])**2

## Fit NPMLE using ebpy
m = GLMixture(prec_type='diagonal', homoscedastic=False)
m.initialize_atoms_grid(X, n_atoms=10000)
m.fit(X, prec, max_iter_em=0, score_every=None,
      n_chunks=10, log_prob_thresh=-8, row_condition=True)

atoms, weights = m.get_params()
spec = 'num_atoms=%s_full_n=%s' % (len(atoms), n)

gmleb = np.vstack([m.posterior_mean(X[chunk], prec[chunk]) for chunk in np.array_split(np.arange(n), 10)])

## Plot raw and denoised CMD on subsample
import matplotlib.pyplot as plt
import random

plt.rcParams.update({'font.size': 20,
                     'mathtext.fontset': 'stix',
                     'font.family': 'STIXGeneral'})


def absMag(almostAbsMag):
    almostAbsMag_in_arcseconds = almostAbsMag/1e3 #first convert parallax from mas ==> arcseconds
    return 5.*np.log10(10.*almostAbsMag_in_arcseconds)


sub = random.sample(range(n), 100000)

fig, ax = plt.subplots(1,2, sharex=True, sharey=True, figsize=(14,7))

ax[0].scatter(X[sub,0], absMag(X[sub,1]), s=1, alpha = 0.025, color='black')
ax[1].scatter(gmleb[sub,0], absMag(gmleb[sub, 1]), s=1, alpha = 0.025, color='blue')
ax[0].set_ylim(6,-6.5)
ax[0].set_xlim(-.3,1.25)

ax[0].text(-.25, -5.5, r'Raw Data') # Observed Values
ax[1].text(-.25, -5.5, r'Empirical Bayes') #GMLEB Estimates

ax[0].set_xlabel('Color (J-K$_s$)$^C$')
ax[1].set_xlabel('Color (J-K$_s$)$^C$')
ax[0].set_ylabel(r'Magnitude M$_\mathrm{J}^C$')

plt.tight_layout()

ax[1].text(.6, -.2, r'Red Clump')
ax[1].text(.8, -3, r'Red Giant Branch')
ax[1].text(.65, 4, r'Binary Sequence')
ax[1].text(.17, 5, 'Lower Main\nSequence')
ax[1].text(-.2, 0, 'Upper Main\nSequence')

plt.savefig('CMD_grid.png')

## Plot initial grid and final NPMLE
rgbs = np.zeros((A,4))
rgbs[:, 0] = 1.0
rgbs[:, 3] = (np.maximum(wghts.flatten(), 0) / np.max(wghts))

fig, ax = plt.subplots(1,2, sharex=True, sharey=True, figsize=(14,7))

ax[0].scatter(atoms[:,0], absMag(atoms[:,1]), marker='o', color='r', s=1, alpha = 0.1)
ax[1].scatter(atoms[:,0], absMag(atoms[:,1]), marker='o', color='r', s=25*rgbs[:, 3])
plt.ylim(6,-6.5)
plt.xlim(-.3,1.25)

ax[0].text(-.25, -5.5, r'Initial Grid', fontsize=18)
ax[1].text(-.25, -5.5, r'Final Estimator $\widehat{G}_n$', fontsize=18)

ax[0].set_xlabel('Color (J-K$_s$)$^C$')
ax[1].set_xlabel('Color (J-K$_s$)$^C$')
ax[0].set_ylabel(r'Magnitude M$_\mathrm{J}^C$')

plt.tight_layout()

plt.savefig('grid_prior.png')
