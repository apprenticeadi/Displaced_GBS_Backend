import strawberryfields as sf
from strawberryfields.ops import *

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

prog = sf.Program(1)

title = 'Squeezed Vacuum'

with prog.context as q:
    if 'Squeezed' in title:
        S = Sgate(1)
        S | q[0]
    if 'Displaced' in title:
        Dgate(-3/np.sqrt(2)) | q[0]
        Rgate(np.pi / 2) | q[0]
        Dgate(3/np.sqrt(2)) | q[0]
        Rgate(-np.pi/2) | q[0]
    else:
        pass


eng = sf.Engine('gaussian')
state = eng.run(prog).state

fig = plt.figure()
X = np.linspace(-10, 10, 100)
P = np.linspace(-10, 10, 100)
Z = state.wigner(0, X, P)
X, P = np.meshgrid(X, P)
ax = fig.add_subplot(111, projection="3d")
ax.set_facecolor('lightgrey')

plt.rcParams['axes.titley'] = 1.0    # y is in axes-relative coordinates.
plt.rcParams['axes.titlepad'] = -14  # pad is in points...
# ax.set_title(title, fontsize=20)

ax.plot_surface(X, P, Z, cmap="RdYlGn", lw=0.5, rstride=1, cstride=1)

# Draw centered axes
val = [12,0,0]
labels = ['x', 'p']
for v in range(len(labels)):
    x = [val[v-0], -val[v-0]]
    y = [val[v-1], -val[v-1]]
    z = [0, 0]
    ax.plot(x,y,z,'k-', linewidth=1)
    ax.text(val[v-0], val[v-1], val[v-2]/60, labels[v], color='k', fontsize=20)

ax.set_axis_off()

fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
fig.set_size_inches(6,6)

fig.savefig(rf'..\Plots\wigner_functions\{title}.png')