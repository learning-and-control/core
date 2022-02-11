from abc import abstractmethod
from torch import cat, matmul, sum
from torch.linalg import solve

from .system_dynamics import SystemDynamics


class RoboticDynamics(SystemDynamics):
  @abstractmethod
  def D(self, qs):
    raise NotImplementedError

  @abstractmethod
  def C(self, qs, q_dots):
    raise NotImplementedError

  @abstractmethod
  def U(self, qs):
    raise NotImplementedError

  @abstractmethod
  def G(self, qs):
    raise NotImplementedError

  @abstractmethod
  def B(self, qs):
    raise NotImplementedError

  @abstractmethod
  def F_ext(self, qs, q_dots):
    raise NotImplementedError

  def T(self, qs, q_dots):
    D_q_dots = matmul(self.D(qs), q_dots.unsqueeze(-1))[..., 0]
    Ts = sum(D_q_dots * q_dots, dim=1) / 2
    return Ts

  def E(self, qs, q_dots):
    Es = self.T(qs, q_dots) + self.U(qs)
    return Es

  def H(self, qs, q_dots):
    Cs = self.C(qs, q_dots)
    Hs = matmul(Cs, q_dots.unsqueeze(-1))[..., 0] + self.G(qs)
    return Hs

  def forward(self, xs, us):
    _, dim = xs.shape
    n = dim // 2
    qs = xs[:, :n]
    q_dots = xs[:, n:]

    forces = matmul(self.B(qs), us.unsqueeze(-1))[..., 0] + self.F_ext(qs, q_dots)
    q_ddots = solve(self.D(qs), forces - self.H(qs, q_dots))
    x_dots = cat([q_dots, q_ddots], dim=1)

    return x_dots
