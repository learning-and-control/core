from numpy import dot
from torch import tensor, float64, from_numpy
from torch.nn import Module, Parameter

from core.dynamics import AffineDynamics
from core.dynamics import LinearizableDynamics
from core.dynamics import SystemDynamics

class LinearSystemDynamics(SystemDynamics, AffineDynamics,
                           LinearizableDynamics, Module):
    """Class for linear dynamics of the form x_dot = A * x + B * u."""

    def __init__(self, A, B):
        """Create a LinearSystemDynamics object.

        Inputs:
        State matrix, A: numpy array
        Input matrix, B: numpy array
        """

        n, m = B.shape
        assert A.shape == (n, n)

        SystemDynamics.__init__(self, n, m)
        Module.__init__(self)
        self.A = Parameter(from_numpy(A).to(float64, copy=True))
        self.B = Parameter(from_numpy(B).to(float64, copy=True))

    def drift(self, x, t):

        return self.A@x

    def act(self, x, t):
        return self.B

    def jacobian_impl(self, x, u, t):
        return self.A, self.B

    def linear_system(self):
        return self.A, self.B