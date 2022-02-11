from torch import cos, ones, sin, zeros

from .robotic_dynamics import RoboticDynamics


class InvertedPendulum(RoboticDynamics):
  def __init__(self, m, l, g=9.81):
    RoboticDynamics.__init__(self)
    self.n = 1
    self.m = 1
    self.params = m, l, g

  def D(self, qs):
    m, l, _ = self.params
    Ds = m * l ** 2 * ones(len(qs), self.n, self.n)
    return Ds

  def C(self, qs, q_dots):
    Cs = zeros(len(qs), self.n, self.n)
    return Cs

  def U(self, qs):
    m, l, g = self.params
    thetas = qs[:, 0]
    Us = m * l * g * cos(thetas)
    return Us

  def G(self, qs):
    m, l, g = self.params
    Gs = -m * g * l * sin(qs)
    return Gs

  def B(self, qs):
    Bs = ones(len(qs), self.n, self.m)
    return Bs

  def F_ext(self, qs, q_dots):
    F_exts = zeros(len(qs), self.n)
    return F_exts
