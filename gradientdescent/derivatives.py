"""List of methods for derivative computation."""
import numpy as np


class NumericalDerivative(object):
    """Compute numerical derivative.

    This class comptes the numerical derivative based on the symmetric
    difference.

    Usage:
        nd = NumericalDerivative(f)
        nd(x) # returns the f' at x
    """

    def __init__(self, f, h=0.0001):
        """Set initial prarameters.

        f: the target function for which to do the calculation.
        h: the difference for interpolating the derivation.
        """
        self.h = h
        self.f = f

    def __call__(self, x):
        """Return value y of y = f'(x)."""
        f, h = self.f, self.h
        return (f(x+h) - f(x-h))/(2.0*h)
