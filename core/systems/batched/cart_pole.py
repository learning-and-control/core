from torch import cos, ones, sin, stack, zeros

from .robotic_dynamics import RoboticDynamics


class CartPole(RoboticDynamics):
  def __init__(self, m_c, m_p, l, g=9.81):
    RoboticDynamics.__init__(self)
    self.params = m_c, m_p, l, g
    self.n = 2
    self.m = 1

  def D(self, qs):
    m_c, m_p, l, _ = self.params
    _, thetas = qs.T

    es = ones(len(qs))
    D_11s = (m_c + m_p) * es
    D_12s = m_p * l * cos(thetas)
    D_22s = m_p * (l ** 2) * es
    D_1s = stack([D_11s, D_12s], dim=1)
    D_2s = stack([D_12s, D_22s], dim=1)
    Ds = stack([D_1s, D_2s], dim=1)
    return Ds

  def C(self, qs, q_dots):
    _, m_p, l, _ = self.params
    _, thetas = qs.T
    _, theta_dots = q_dots.T

    es = zeros(len(qs))
    C_11s = es
    C_12s = -m_p * l * theta_dots * sin(thetas)
    C_21s = es
    C_22s = es
    C_1s = stack([C_11s, C_12s], dim=1)
    C_2s = stack([C_21s, C_22s], dim=1)
    Cs = stack([C_1s, C_2s], dim=1)
    return Cs

  def U(self, qs):
    _, m_p, l, g = self.params
    _, thetas = qs.T
    Us = m_p * g * l * cos(thetas)
    return Us

  def G(self, qs):
    _, m_p, l, g = self.params
    _, thetas = qs.T

    es = zeros(len(qs))
    G_1s = es
    G_2s = -m_p * g * l * sin(thetas)
    Gs = stack([G_1s, G_2s]).T
    return Gs

  def B(self, qs):
    N = len(qs)
    e_0s = zeros(N, 1)
    e_1s = ones(N, 1)
    Bs = stack([e_1s, e_0s], dim=1)
    return Bs

  def F_ext(self, qs, q_dots):
    F_exts = zeros(len(qs), self.n)
    return F_exts
