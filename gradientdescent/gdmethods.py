"""Gradient methods."""
import collections

import numpy as np

from derivatives import NumericalDerivative


class GradientMethodUnivariate(collections.Iterator):
    """Set up the gradient descent method."""

    def __init__(self, x0, targetFunction, derivative = None, **kwargs):
        """Set up the gradient method prarameters.

        Arguments
            targetFunction : the primary function for minimization
            derivative     : a *callable* function that computes the tangent
                             at a given x
        """
        self.targetFunction = targetFunction
        if derivative is None:
            self.derivative = NumericalDerivative(targetFunction)

        if 'maxiter' in kwargs:
            self.maxiter = kwargs['maxiter']
        else:
            self.maxiter = 1000

        self.x0 = x0
        # step initialization
        self.x = x0
        self.y = self.targetFunction(self.x0)
        self._current_iteration = 0

    def reset(self):
        raise NotImplementedError('The child class should implement this method.')

    def terminalCondition(self):
        raise NotImplementedError('The child class should implement this method.')

    def _reset(self):
        self._current_iteration = 0
        self.x = self.x0
        self.y = self.targetFunction(self.x)

    def nextStep(self):
        """Compute next iteration step.

        The nextStep method should return the new x. y will be implicitly
        computed from the new x.
        """
        raise NotImplementedError('The child class should implement this method.')

    def isDone(self):
        """Compute the terminal condition.

        Terminate if y and ynew are close to each other or we reached the
        upper limit of iterations. Further terminal condition can
        be implemented.
        """
        isclose = np.isclose(self.y, self._ynew, rtol=1e-08, atol=1e-12)
        maxIterations = (self._current_iteration > self.maxiter)
        return (isclose or maxIterations or self.terminalCondition())

    def __iter__(self):
        self._reset()
        self.reset()
        return self

    def next(self):
        self._xnew = self.nextStep()
        self._ynew = self.targetFunction(self._xnew)
        if self.isDone():
            raise StopIteration
        else:
            self._current_iteration += 1
            self.x = self._xnew
            self.y = self._ynew
            return self._xnew, self._ynew

    def tangent(self, xlim):
        x = np.asarray(xlim)
        return self.derivative(self.x) * (x - self.x) + self.y


class GradientMethodNewton(GradientMethodUnivariate):
    """Compute gradient descent via Newton's method."""

    def __init__(self, *args, **kwargs):
        GradientMethodUnivariate.__init__(self, *args, **kwargs)

    def terminalCondition(self):
        return False

    def reset(self):
        pass

    def nextStep(self):
        """Compute and return new x. y will be implicitly computed."""
        dx = self.derivative(self.x)
        # Newton's method
        xn = self.x - (self.y / dx)
        return xn


class GradientMethodSimple(GradientMethodUnivariate):
    """Compute gradient descent via simple method using fixed mu."""

    def __init__(self, mu = None, mu_decay = None, *args, **kwargs):
        GradientMethodUnivariate.__init__(self, *args, **kwargs)
        if mu is None:
            self.mu0 = 0.1
        else:
            self.mu0 = mu

        if mu_decay is None:
            self.mu_decay = lambda x: x.mu0
        else:
            self.mu_decay = mu_decay

    def terminalCondition(self):
        return False

    def nextStep(self):
        """Compute and return new x. y will be implicitly computed."""
        dx = self.derivative(self.x)
        # GD update
        xn = self.x - (self.mu * dx)
        self.mu = self.mu_decay(self)
        return xn

    def reset(self):
        self.mu = self.mu0


class GradientMethodMomentum(GradientMethodUnivariate):
    """Compute gradient descent via simple method using fixed mu."""

    def __init__(self, friction = 0.9, mu = None, mu_decay = None, *args, **kwargs):
        GradientMethodUnivariate.__init__(self, *args, **kwargs)
        if mu is None:
            self.mu0 = 0.1
        else:
            self.mu0 = mu

        if mu_decay is None:
            self.mu_decay = lambda x: x.mu0
        else:
            self.mu_decay = mu_decay

        self.v = 0
        self.friction = friction

    def terminalCondition(self):
        return False

    def nextStep(self):
        """Compute and return new x. y will be implicitly computed."""
        dx = self.derivative(self.x)
        # GD update
        self.v = self.friction * self.v - (self.mu * dx)
        xn = self.x + self.v
        self.mu = self.mu_decay(self)
        return xn

    def reset(self):
        self.mu = self.mu0
        self.v = 0


class GradientMethodNesterov(GradientMethodMomentum):
    """Compute gradient descent via Nesterov's Accelerated Momentum (NAG)."""

    def __init__(self, *args, **kwargs):
        GradientMethodMomentum.__init__(self, *args, **kwargs)

    def nextStep(self):
        """Compute and return new x. y will be implicitly computed."""
        x_future = self.x + (self.friction * self.v)
        dx = self.derivative(x_future)
        # GD update
        self.v = (self.friction * self.v) - (self.mu * dx)
        xn = self.x + self.v
        self.mu = self.mu_decay(self)
        return xn


class GradientMethodADAM(GradientMethodMomentum):
    """Compute gradient descent via adaptive moment estimation."""

    def __init__(self, beta1 = 0.9, beta2 = 0.999, eps = 1e-8, *args, **kwargs):
        GradientMethodMomentum.__init__(self, *args, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = 0

    def nextStep(self):
        """Compute and return new x. y will be implicitly computed."""
        print '##### iteration %d #####' % self._current_iteration
        itercnt = self._current_iteration + 1
        dx = self.derivative(self.x)
        print 'x: %f, dx: %f' % (self.x, dx)
        self.m = self.beta1 * self.m + ((1 - self.beta1) * dx)
        mt = self.m / (1 - self.beta1 ** itercnt)
        print 'm: %f, mt: %f' % (self.m, mt)
        self.v = self.beta2 * self.v + (1 - self.beta2) * (dx**2)

        vt = self.v / (1 - self.beta2 ** itercnt)
        print 'v: %f, vt: %f' % (self.v, vt)
        xn = self.x - self.mu * mt / (np.sqrt(vt) + self.eps)
        print 'new x: %f' % xn
        self.mu = self.mu_decay(self)
        return xn

    def reset(self):
        self.mu = self.mu0
        self.v = 0
        self.m = 0
