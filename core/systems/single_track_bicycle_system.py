from torch import cat, cos, diag, float64, norm, sin, stack, tensor, zeros, set_default_dtype
from torch import atan2, atan, isnan
from torch.nn import Module, Parameter

from core.dynamics import SystemDynamics, AffineDynamics
from core.util import  default_fig
from numpy import pi
set_default_dtype(float64)

# Bicycle Dynamics from here:
# https://github.com/urosolia/RacingLMPC/blob/master/src/fnc/SysModel.py#L130
# TODO: Ask Ugo Whats up with this Curvature Stuff
# https://github.com/urosolia/RacingLMPC/blob/master/src/fnc/Utilities.py

# Vehicle Parameters
class SingleTrackBicycle(SystemDynamics, AffineDynamics, Module):
    def __init__(self):
        SystemDynamics.__init__(self, 6, 2)
        Module.__init__(self)

    def forward(self, x, u, t):
        m = 1.98
        lf = 0.125
        lr = 0.125
        Iz = 0.024
        Df = 0.8 * m * 9.81 / 2.0
        Cf = 1.25
        Bf = 1.0
        Dr = 0.8 * m * 9.81 / 2.0
        Cr = 1.25
        Br = 1.0

        a = u[0]
        delta = u[1]

        X = x[0]
        Y = x[1]
        w = x[2]

        vx = x[3]
        vy = x[4]
        wz = x[5]

        # Compute tire split angle
        alpha_f = delta - atan2(vy + lf * wz, vx)
        alpha_r = - atan2(vy - lf * wz, vx)

        # Compute lateral force at front and rear tire
        Fyf = Df * sin(Cf * atan(Bf * alpha_f))
        Fyr = Dr * sin(Cr * atan(Br * alpha_r))

        # Propagate the dynamics of deltaT
        x_dot = cat([
        ((vx * cos(w) - vy * sin(w)))[None],
        (vx * sin(w) + vy * cos(w))[None],
        wz[None],
        (a - 1 / m * Fyf * sin(delta) + wz * vy)[None],
        (1 / m * (Fyf * cos(delta) + Fyr) - wz * vx)[None],
        (1 / Iz * (lf * Fyf * cos(delta) - lr * Fyr))[None]], dim=0)
        return x_dot