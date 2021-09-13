##### Convex hull counterexample (Figure 4) #####

## public github repository: 
## https://github.com/jake-soloff/ebpy
from ebpy.GLMixture import GLMixture, mvn_pdf

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

## Define data and precision matrices
X = np.array([[0, 1], 
              [0, -1],
              [1, 0],
              [-1, 0]])
a, b = .2, 20
prec = np.array([[a, b],
                 [a, b],
                 [b, a],
                 [b, a]])
n, d = X.shape

## Define grid of atoms (note all diagonal covariance matrices)
M = 1
B = 50
g = np.linspace(-M, M, B)
atoms = np.stack(np.meshgrid(g,g)).reshape((d, B**2)).T

## Fit NPMLE using ebpy
m = GLMixture(prec_type='diagonal', homoscedastic=False, atoms_init=atoms)
_ = m.fit(X, prec=prec, max_iter_em=2000)

## Plot raw data, NPMLE atoms and covariance matrices
plt.rcParams.update({'font.size':22})
fig, ax = plt.subplots(figsize=(8,8))
ax.scatter(X[:,0], X[:,1],label='Data', c='blue')

for i in range(n):
    e = Ellipse(X[i], 1/np.sqrt(prec[i,0]), 1/np.sqrt(prec[i,1]), linewidth=2, fill=False, ls='--')
    ax.add_patch(e)
    ax.text(X[i,0] + (.05 if i < 2 else -.05), X[i, 1] + (-.05 if i < 2 else .05), '$X_{%d}$' %(i+1), color='blue') # +.02 -.1
e = Ellipse(X[i], 1/np.sqrt(prec[i,0]), 1/np.sqrt(prec[i,1]), linewidth=2, fill=False, label='Covariance', ls='--')
ax.add_patch(e)
    
j = 0
ax.scatter(m.atoms[j, 0], m.atoms[j, 1], alpha=m.weights[j]**.05, c='r',label='Atoms')
for i, j in enumerate(np.where(m.weights > 0.1)[0]):#range(1, len(m.weights)):
    ax.scatter(m.atoms[j, 0], m.atoms[j, 1], c='r')
    ax.text((m.atoms[j, 0]-.1)*.8, (m.atoms[j, 1]-.05)*.82, '$a_{%d}$' %(i+1), color='r')

plt.xticks([-1, 0, 1])
plt.yticks([-1, 0, 1])
plt.savefig('convex_hull_counterexample.png')

## Make 3D contour plot
k = 3000 
K = 2*k+1
xx = np.linspace(-1.5, 1.5, K)
yy = np.linspace(-1.5, 1.5, K)

(XX, YY) = np.meshgrid(xx, yy)

ZZ = np.mean([np.exp(-(prec[i, 0]/2) * (XX-X[i, 0])**2)*np.exp(-(prec[i, 1]/2) * (YY-X[i, 1])**2) for i in range(4)], axis=0)

fig = plt.figure(figsize=(15,10))
ax = fig.gca(projection='3d')

# Plot the surface.
surf = ax.plot_surface(XX, YY, ZZ, cmap='Purples',
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-.01, 0.5)
ax.set_zticks([])

plt.savefig('mixture3d.png')
