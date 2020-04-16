from torch import matmul

from .dynamics import Dynamics
from ..util import torch_guard


class AffineDynamics(Dynamics):
    """Abstract class for dynamics of the form x_dot = f(x, t) + g(x, t) * u.

    Override eval, drift, act.
    """

    def drift(self, x, t):
        """Compute drift vector f(x, t).

        Inputs:
        State, x: numpy array
        Time, t: float

        Outputs:
        Drift vector: numpy array
        """

        return torch_guard((x, t), self.drift_impl)

    def drift_impl(self, x, t):
        pass

    def act(self, x, t):
        """Compute actuation matrix g(x, t).

        Inputs:
        State, x: numpy array
        Time, t: float

        Outputs:
        Actuation matrix: numpy array
        """

        return torch_guard((x, t), self.act_impl)

    def act_impl(self, x, t):
        pass

    def eval_dot_impl(self, x, u, t):
        return self.drift(x, t) + matmul(self.act(x, t), u)