import numpy as np
from numpy import dot
from torch import tensor, float64, from_numpy
from torch.nn import Module, Parameter

from core.dynamics import AffineDynamics, LinearizableDynamics, SystemDynamics, ObservableDynamics


class LinearSystemDynamics(SystemDynamics,
                           AffineDynamics,
                           LinearizableDynamics,
                           ObservableDynamics,
                           Module):
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
        ObservableDynamics.__init__(self, obs_dim=n)
        Module.__init__(self)
        self.register_parameter('A', Parameter(from_numpy(A).to(float64, copy=True)))
        self.register_parameter('B', Parameter(from_numpy(B).to(float64, copy=True)))

    def drift(self, x, t):
        #A in nxn
        #x is bxn
        return (self.A[None] @ x[:, :, None])[..., 0]

    def act(self, x, t):
        return self.B

    def jacobian_impl(self, x, u, t):
        n_batch = x.shape[0]
        return self.A[None].expand(n_batch, -1, -1), \
               self.B[None].expand(n_batch,-1, -1)


    def linear_system(self):
        return self.A, self.B

    def get_observation(self, state):
        return state

    def to_principal_coordinates(self, state):
        return state

class NIntegrator(LinearSystemDynamics):

    def __init__(self, n):
        A = np.eye(n)
        A[-1, :] = 0.
        B = np.zeros((n, 1))
        B[-1, 0] = 1.
        LinearSystemDynamics.__init__(self, A, B)

class RandomLinearSystem(LinearSystemDynamics):

    def __init__(self, n, m):
        A = np.random.randn(n, n)
        B = np.random.randn(n, m)
        LinearSystemDynamics.__init__(self, A, B)
