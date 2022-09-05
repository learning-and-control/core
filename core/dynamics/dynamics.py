from torch.autograd.functional import jacobian
from torch import eye, matrix_exp, pinverse, stack
from torch.nn import Module

class Dynamics(Module):
    """Abstract dynamics class.

    Override image, forward.
    """
    def __init__(self):
        super().__init__()

    def image(self, x, t):
        """Compute representation.

        Inputs:
        State, x: numpy array
        Time, t: float

        Outputs:
        Representation: numpy array
        """

        pass

    def forward(self, x, u, t):
        """Compute dynamics (time derivative of representation).

        Inputs:
        State, x: numpy array
        Action, u: numpy array
        Time, t: float

        Outputs:
        Time-derivative: numpy array
        """

        pass

    def jacobian(self, x, u, t):

        F,G = jacobian(
            lambda x_in, u_in: self(x_in, u_in, t).sum(dim=0),
            (x, u), create_graph=True)
        F = F.swapaxes(1,0)
        G = G.swapaxes(1,0)
        return F, G

    def jacobian_exp(self, x, u, t, delta_t):
        ndims = x.ndim
        unsqueezed = False
        if ndims == 1:
            unsqueezed = True
            assert x.ndim == u.ndim
            x = x.unsqueeze(0)
            u = u.unsqueeze(0)
        expAs = list()
        expBs = list()
        for i in range(x.shape[0]):
            A, B = jacobian(self, (x[i], u[i], t))[:-1]
            expA = matrix_exp(A*delta_t)
            expB = pinverse(A) @ (expA - eye(self.n)) @ B
            expAs.append(expA)
            expBs.append(expB)
        if unsqueezed:
            return stack(expAs, dim=0)[0], stack(expBs, dim=0)[0]
        else:
            return stack(expAs, dim=0), stack(expBs, dim=0)