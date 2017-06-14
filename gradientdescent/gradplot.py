"""Plot methods for gradient descent."""
import numpy as np


class GMAnimator(object):
    """Animate gradient evolution."""

    def __init__(self, ax, grmethod, fromx = -1, tox = 1, by = 0.01):
        self.grmethod = grmethod
        self._ax = ax
        self._currentstep, = self._ax.plot([], [], '.r')
        self._tangentline, = self._ax.plot([], [], '-g')
        xr = np.arange(fromx, tox, by)
        yr = self.grmethod.targetFunction(xr)
        self._graph = self._ax.plot(xr, yr, '-b')

    def reset(self):
        self._currentstep.set_data([], [])
        self._tangentline.set_data([], [])
        self.grmethod.reset()
        return self._currentstep,

    def __call__(self, i):
        print 'a', i
        try:
            xlim = self._ax.get_xlim()
            ylim = self.grmethod.tangent(xlim)
            self._tangentline.set_data(xlim, ylim)
            self._currentstep.set_data(self.grmethod._xnew, self.grmethod._ynew)
        except Exception as e:
            print e

        return self._currentstep, self._tangentline
