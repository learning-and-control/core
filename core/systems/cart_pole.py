from torch import cat, cos, float64, sin, stack, tensor, zeros_like, ones_like, atan2
from torch.nn import Module, Parameter
from core.dynamics import RoboticDynamics, AffineDynamics, ObservableDynamics

class CartPole(RoboticDynamics, ObservableDynamics):
    '''
    Cart-pole system
        #q = (x, theta)
        #q_dot = (x_dot, theta_dot)
        #state = (x, theta, x_dot, theta_dot) = (q, q_dot)
    '''
    def __init__(self, m_c, m_p, l, g=9.81):
        RoboticDynamics.__init__(self, 2, 1)
        ObservableDynamics.__init__(self, 5)
        self.params = Parameter(tensor([m_c, m_p, l, g], dtype=float64))

    def get_observation(self, state):
        x = state[..., 0]
        theta = state[..., 1]
        x_dot = state[..., 2]
        theta_dot = state[..., 3]
        return stack([
            x,
            cos(theta),
            sin(theta),
            x_dot,
            theta_dot
        ], dim=-1)
    def to_principal_coordinates(self, state):
        x = state[..., 0]
        theta = state[..., 1]
        x_dot = state[..., 2]
        theta_dot = state[..., 3]
        p_theta = atan2(sin(theta), cos(theta)) #y, x
        return stack([
            x,
            p_theta,
            x_dot,
            theta_dot
        ], dim=-1)

    def forward(self, x, u, t):
        m_c, m_p, l, g = self.params
        q = x[:, :2]
        q_dot = x[:, 2:]

        sin_th = sin(q[:,1])
        cos_th = cos(q[:,1])
        th_dot = q_dot[:,1]
        u = u[:,0]
        accel = (u + m_p*sin_th*(l*(th_dot**2) + g*cos_th))/(m_c + m_p * (sin_th**2))
        thddot = (-u*cos_th - m_p*l*(th_dot**2)*cos_th*sin_th - (m_c+m_p)*g*sin_th)/(l*(m_c + m_p*(sin_th**2)))
        xddot = stack([accel, thddot], dim=1)
        return cat([q_dot, xddot], dim=1)

    def D(self, q):
        m_c, m_p, l, _ = self.params
        theta = q[:,1]
        D_row_1 = stack([m_c + m_p * ones_like(theta), m_p * l * cos(theta)], dim=1)
        D_row_2 = stack([m_p * l * cos(theta), ones_like(theta) * m_p * (l ** 2)],dim=1)
        D = stack((D_row_1, D_row_2),dim=1)
        return D

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
        return stack([z, m_p * g * l * sin(theta)], dim=1)

    def B(self, q):
        return tensor([[[1], [0]]], dtype=float64, device=q.device
                      ).expand(q.shape[0], -1, -1)
