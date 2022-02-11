from torch import cos, eye, ones, sin, stack, zeros

from .robotic_dynamics import RoboticDynamics


class DoubleInvertedPendulum(RoboticDynamics):
    def __init__(self, m_1, m_2, l_1, l_2, g=9.81):
        RoboticDynamics.__init__(self)
        self.n = 2
        self.m = 2
        self.params = m_1, m_2, l_1, l_2, g

    def D(self, qs):
        m_1, m_2, l_1, l_2, _ = self.params
        _, theta_2s = qs.T

        D_11s = (m_1 + m_2) * (l_1 ** 2) + 2 * m_2 * l_1 * l_2 * cos(theta_2s) + m_2 * (l_2 ** 2)
        D_xs = m_2 * l_1 * l_2 * cos(theta_2s) + m_2 * (l_2 ** 2)
        D_1s = stack([D_11s, D_xs], dim=1)

        D_22s = m_2 * (l_2 ** 2) * ones(len(D_xs))
        D_2s = stack([D_xs, D_22s], dim=1)

        Ds = stack([D_1s, D_2s], dim=1)
        return Ds

    def C(self, qs, q_dots):
        _, m_2, l_1, l_2, _ = self.params
        _, theta_2s = qs.T
        theta_1_dots, theta_2_dots = q_dots.T

        C_11s = zeros(len(qs))
        C_12s = -m_2 * l_1 * l_2 * (2 * theta_1_dots + theta_2_dots) * sin(theta_2s)
        C_1s = stack([C_11s, C_12s], dim=1)

        C_21s = -C_12s / 2
        C_22s = -m_2 * l_1 * l_2 * theta_1_dots * sin(theta_2s) / 2
        C_2s = stack([C_21s, C_22s], dim=1)

        Cs = stack([C_1s, C_2s], dim=1)
        return Cs

    def U(self, qs):
        m_1, m_2, l_1, l_2, g = self.params
        theta_1s, theta_2s = qs.T
        Us = (m_1 + m_2) * g * l_1 * cos(theta_1s) + m_2 * g * l_2 * cos(theta_1s + theta_2s)
        return Us

    def G(self, qs):
        m_1, m_2, l_1, l_2, g = self.params
        theta_1s, theta_2s = qs.T
        G_1s = -(m_1 + m_2) * g * l_1 * sin(theta_1s) - m_2 * g * l_2 * sin(theta_1s + theta_2s)
        G_2s = -m_2 * g * l_2 * sin(theta_1s + theta_2s)
        Gs = stack([G_1s, G_2s], dim=1)
        return Gs

    def B(self, qs):
        Bs = eye(2).unsqueeze(0).repeat(len(qs), 1, 1)
        return Bs

    def F_ext(self, qs, q_dots):
        return zeros(len(qs), 2)
