from ..util import torch_guard
from torch.autograd.functional import jacobian
from torch import eye, matrix_exp, pinverse, stack

class Dynamics:
    """Abstract dynamics class.

    Override eval, eval_dot.
    """

    def eval(self, x, t):
        """Compute representation.

        Inputs:
        State, x: numpy array
        Time, t: float

        Outputs:
        Representation: numpy array
        """

        pass

    def eval_dot(self, x, u, t):
        """Compute dynamics (time derivative of representation).

        Inputs:
        State, x: numpy array
        Action, u: numpy array
        Time, t: float

        Outputs:
        Time-derivative: numpy array
        """

        return torch_guard((x, u, t), self.eval_dot_impl)

    def eval_dot_impl(self, x, u, t):
        pass

    def jacobian(self, x, u, t):
        return torch_guard((x, u, t), self.jacobian_impl)

    def jacobian_impl(self, x, u, t):
        return jacobian(self.eval_dot, (x, u, t))[:-1]

    def jacobian_exp(self, x, u, t, delta_t):
        return torch_guard((x, u, t, delta_t), self.jacobian_exp_impl)

    def jacobian_exp_impl(self, x, u, t, delta_t):
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
            A, B = jacobian(self.eval_dot, (x[i], u[i], t))[:-1]
            expA = matrix_exp(A*delta_t)
            expB = pinverse(A) @ (expA - eye(self.n)) @ B
            expAs.append(expA)
            expBs.append(expB)
        if unsqueezed:
            return stack(expAs, dim=0)[0], stack(expBs, dim=0)[0]
        else:
            return stack(expAs, dim=0), stack(expBs, dim=0)