from torch import cat, cos, diag, float64, norm, sin, stack, tensor, zeros, set_default_dtype
from torch import atan2
from torch.nn import Module, Parameter
from core.dynamics import SystemDynamics, AffineDynamics

set_default_dtype(float64)


# Car Dynamics Copied from here:
# https://github.com/urosolia/RacingLMPC/blob/master/src/fnc/SysModel.py#L130

class Car(SystemDynamics, AffineDynamics, Module):

    def __init__(self):
        super(SystemDynamics, self).__init__(6, 2)
        super(Module, self).__init__()
        self.m  = Parameter(tensor(1.98))
        self.lf = Parameter(tensor(0.125))
        self.lr = Parameter(tensor(0.125))
        self.Iz = Parameter(tensor(0.024))
        self.Cf = Parameter(tensor(1.25))
        self.Bf = Parameter(tensor(1.0))
        self.Cr = Parameter(tensor(1.25))
        self.Br = Parameter(tensor(1.0))

        self.Df = 0.8 * self.m * 9.81 / 2.0
        self.Dr = 0.8 * self.m * 9.81 / 2.0

    def eval_dot_impl(self, x, u, t):
        delta = u[:,0]
        a = u[:, 1]
        X = Y = psi = tensor(0.0) #assume these are for global coordinate frame
        vx   = x[:,0]
        vy   = x[:,1]
        wz   = x[:,2]
        epsi = x[:,3]
        s    = x[:,4]
        ey   = x[:,5]

        # Compute tire split angle
        alpha_f = self.delta - atan2(vy + self.lf * wz, vx)
        alpha_r = - atan2(vy - self.lf * wz, vx)

        # Compute lateral force at front and rear tire
        Fyf = self.Df * sin(self.Cf * atan2(self.Bf * alpha_f))
        Fyr = self.Dr * sin(self.Cr * atan2(self.Br * alpha_r))
        return stack([
            vx + self.deltaT * (a - 1 / self.m * Fyf * sin(delta) + wz * vy),
            vy + self.deltaT * (1 / self.m * (Fyf * cos(delta) + Fyr) - wz * vx),
            wz + self.deltaT * (1 / self.Iz * (self.lf * Fyf * cos(delta) - self.lr * Fyr)),
            psi + self.deltaT * (wz),
            X + self.deltaT * (vx * cos(psi) - vy * sin(psi)),
            Y + self.deltaT * (vx * sin(psi) + vy * cos(psi)),
        ], dim=1)