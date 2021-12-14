
from torch import cat, cos, diag, float64, norm, sin, stack, tensor, zeros
from torch.nn import Module, Parameter
from core.dynamics import RoboticDynamics

class Unicycle(RoboticDynamics, Module):

    def __init__(self, m, J):
        RoboticDynamics.__init__(self, 3, 2)
        Module.__init__(self)
        self.params = Parameter(tensor([m, J], dtype=float64))

    def D(self, q):
        m, J = self.params
        return diag(stack([m, m, J]))

    def C(self, q, q_dot):
        m, _ = self.params
        x, y, theta = q
        v = norm(stack([x, y]), 2)
        z = tensor(0, dtype=float64)
        return stack([
            stack([z, z, m * v * sin(theta)]),
            stack([z, z, -m * v * cos(theta)]),
            stack([z, z, z])])

    def B(self, q):
        _, _, theta = q
        z = tensor(0, dtype=float64)
        one_ = tensor(1, dtype=float64)
        return stack([
            stack([cos(theta), z]),
            stack([sin(theta), z]),
            stack([z, one_])
        ])

    def G(self, q):
        return zeros(3)

    def get_state_names(self):
        return ['$x$ (m)', '$y$ (m)', '$\\theta$ (rad)$',
                '$\\dot{x}$ (m)', '$\\dot{y}$ (m)', '$\\dot{\\theta}$ (rad)$']
