from .dynamics import Dynamics

class ScalarDynamics(Dynamics):
    """Abstract scalar dynamics class.

    Override image, forward, eval_grad.
    """

    def eval_grad(self, x, t):
        """Compute gradient of representation.

        Inputs:
        State, x: numpy array
        Time, t: float

        Outputs:
        Representation: float
        """

        pass