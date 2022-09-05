
from .controller import Controller

class LinearController(Controller):
    """Class for linear policies."""

    def __init__(self, affine_dynamics, K):
        """Create a LinearController object.

        Policy is u = -K * x.

        Inputs:
        Affine dynamics, affine_dynamics: AffineDynamics
        Gain matrix, K: numpy array
        """

        Controller.__init__(self, affine_dynamics)
        self.register_buffer('K', K)

    def forward(self, x, t):
        return (-self.K[None] @ self.dynamics.image(x, t)[:, :, None])[:, :, 0]
