from torch import cat, cos, float64, sin, stack, tensor, zeros_like, ones_like
from torch.nn import Module, Parameter
from core.dynamics import RoboticDynamics



class CartPole(RoboticDynamics, Module):
    def __init__(self, m_c, m_p, l, g=9.81):
        RoboticDynamics.__init__(self, 2, 1)
        Module.__init__(self)
        self.params = Parameter(tensor([m_c, m_p, l, g], dtype=float64))

    def D(self, q):
        m_c, m_p, l, _ = self.params
        theta = q[:,1]
        return stack(
            (stack([m_c + m_p * ones_like(theta), m_p * l * cos(theta)], dim=1),
             stack([m_p * l * cos(theta), ones_like(theta)*m_p * (l ** 2)], dim=1)), dim=1)

    def C(self, q, q_dot):
        _, m_p, l, _ = self.params
        theta = q[:,1]
        z = zeros_like(theta)
        theta_dot = q_dot[:, 1]
        return stack((stack([z, -m_p * l * theta_dot * sin(theta)], dim=1),
                      stack([z, z], dim=1)), dim=1)

    def U(self, q):
        _, m_p, l, g = self.params
        theta = q[:,1]
        return m_p * g * l * cos(theta)

    def G(self, q):
        _, m_p, l, g = self.params
        theta = q[:,1]
        z = zeros_like(theta)
        return stack([z, -m_p * g * l * sin(theta)], dim=1)

    def B(self, q):
        return tensor([[[1], [0]]], dtype=float64, device=q.device
                      ).expand(q.shape[0], -1, -1)
