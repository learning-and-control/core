from torch import cat, cos, float64, sin, stack, tensor
from torch.nn import Module, Parameter
from core.dynamics import RoboticDynamics



class CartPole(RoboticDynamics, Module):
    def __init__(self, m_c, m_p, l, g=9.81):
        RoboticDynamics.__init__(self, 2, 1)
        Module.__init__(self)
        self.params = Parameter(tensor([m_c, m_p, l, g], dtype=float64))

    def D(self, q):
        m_c, m_p, l, _ = self.params
        _, theta = q
        return stack(
            (stack([m_c + m_p, m_p * l * cos(theta)]),
             stack([m_p * l * cos(theta), m_p * (l ** 2)])))

    def C(self, q, q_dot):
        _, m_p, l, _ = self.params
        z = tensor(0, dtype=float64)
        _, theta = q
        _, theta_dot = q_dot
        return stack((stack([z, -m_p * l * theta_dot * sin(theta)]),
                      stack([z, z])))

    def U(self, q):
        _, m_p, l, g = self.params
        _, theta = q
        return m_p * g * l * cos(theta)

    def G(self, q):
        _, m_p, l, g = self.params
        _, theta = q
        z = tensor(0, dtype=float64)
        return stack([z, -m_p * g * l * sin(theta)])

    def B(self, q):
        return tensor([[1], [0]], dtype=float64)
