"""Example of Gradient Descent."""
import os
from math import cos, sin

import matplotlib.pyplot as plt
import numpy as np
import seaborn
from matplotlib import rc
from matplotlib.animation import FuncAnimation

import gradientdescent.gdmethods as gdms
import gradientdescent.gradplot as gdplt

# print os.getcwd()
rc('animation', html='html5')


def any_function(x):
    """Compute f(x)."""
    # return x**2
    return (x)**2 + 3*sin(x) - 4*cos((x)**2)


any_function = np.vectorize(any_function)

# gd = gdms.GradientMethodNewton(3.1, any_function)
# gd = gdms.GradientMethodSimple(3.1, any_function, mu = lambda x: x.mu)
gd = gdms.GradientMethodSimple(mu = 0.9,
                               mu_decay = lambda x: x.mu0 * np.exp(-0.1 * x._current_iteration),
                               x0 = 3.1,
                               targetFunction = any_function)

for k in gd:
    print k
gd.mu
gd._current_iteration

gd.reset()
fig, ax = plt.subplots()
ax.grid(True)
gma = gdplt.GMAnimator(ax, grmethod = gd, fromx = -6, tox = 6)

anim = FuncAnimation(fig, gma, frames = gd, init_func=gma.reset,
                     interval=300)
# anim.save('movie.mp4')

anim
plt.show()

gd._ynew
gd.y
