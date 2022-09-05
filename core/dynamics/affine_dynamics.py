from torch import matmul

from .dynamics import Dynamics

class AffineDynamics(Dynamics):
    """Abstract class for dynamics of the form x_dot = f(x, t) + g(x, t) * u.

    Override image, drift, act.
    """

    def drift(self, x, t):
        """Compute drift vector f(x, t).

        Inputs:
        State, x: numpy array
        Time, t: float

        Outputs:
        Drift vector: numpy array
        """
        pass

    def act(self, x, t):
        """Compute actuation matrix g(x, t).

        Inputs:
        State, x: numpy array
        Time, t: float

        Outputs:
        Actuation matrix: numpy array
        """

        pass

    def forward(self, x, u, t):
        return self.drift(x, t) + (self.act(x,t)@u[:,:,None]).squeeze(-1)