##### Non-uniqueness (Figure 3) #####

import pickle as pkl
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

## Observations
X = np.array([[0, 1],
              [np.sqrt(.75), -.5],
              [-np.sqrt(.75), -.5]])

## Precision matrices (diagonal & homoscedastic)
s = np.sqrt(3/np.log(256)) 
prec = np.ones(2)/s**2

## Compute dual mixture density over a grid
k = 1000
K = 2*k+1
xx = np.linspace(-1, 1, K)
yy = np.linspace(-1.5, 1.5, K)

XX, YY = np.meshgrid(xx, yy)
ZZ = np.mean((stats.norm.pdf(XX[np.newaxis,:,:] - X[:, 0].reshape((3,1,1)), scale=s)
             *stats.norm.pdf(YY[np.newaxis,:,:] - X[:, 1].reshape((3,1,1)), scale=s)), axis=0)
ZZ -= np.min(ZZ)
ZZ /= np.max(ZZ)

## Plot level sets of dual mixture density, along with modes and observations
plt.figure(figsize=(8,8))
plt.rcParams.update({'font.size':22,
                     'mathtext.fontset': 'stix',  
                     'font.family': 'STIXGeneral'})

levels = np.sort(np.append(np.percentile(ZZ.flatten(), 
                                         q=list(range(0,99,9)) + [99, 100]), 
                           ZZ[k, k-15]))

contours = plt.contour(XX, YY, ZZ, levels, colors='black')


plt.plot(X[:, 0], X[:, 1], 'b.', markersize=15)

plt.text(X[0, 0], X[0, 1]-.09, '$X_1$', c='b')
plt.text(X[0, 0]/2-.06, X[0, 1]/2+.1, '$X_1/2$', c='b')
plt.text(X[1, 0]-.13, X[1, 1]-.03, '$X_2$', c='b')
plt.text(X[1, 0]/2+.05, X[1, 1]/2-.11, '$X_2/2$', c='b')
plt.text(X[2, 0]+.03, X[2, 1]-.03, '$X_3$', c='b')
plt.text(X[2, 0]/2-.15, X[2, 1]/2-.11, '$X_3/2$', c='b')

plt.plot(X[:, 0]/2, X[:, 1]/2, 'b.', markersize=15)
plt.plot(0,0, 'b.', markersize=15)
plt.text(.03, -.05, '$0$', c='b')
plt.xlim([-1, 1])
plt.ylim([-.75, 1.25])
plt.xticks([-1, 0, 1])
plt.yticks([-.5, 0, .5, 1])
plt.tight_layout()
plt.savefig('contours.png')
