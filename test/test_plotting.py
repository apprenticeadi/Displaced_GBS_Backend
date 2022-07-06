import matplotlib.pyplot as plt
import numpy as np


# How to plot on two separate plots in the same for loop, and how to control the color cycler

max_x = 20
x_list = list(range(max_x))

fig1, ax1 = plt.subplots()
ax1.set_title('title 0 ')
ax1.set_xlabel('x 0')
ax1.set_ylabel('y 0')
color_cycler = fig1.gca()._get_lines.prop_cycler

fig2, ax2 = plt.subplots()
ax2.set_title('title 1')
ax2.set_xlabel('x 0')
ax2.set_ylabel('y 0')

for M in range(4):

    color = next(color_cycler)['color']
    ax1.plot(x_list, np.random.rand(max_x), label='M={}'.format(M), color=color)
    ax1.plot(x_list, [M]*max_x, linestyle='--', color=color)

    ax2.plot(x_list, np.random.rand(max_x), label='M={}'.format(M))

ax1.legend()
ax2.legend()