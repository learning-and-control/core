from numpy import array

from core.dynamics import FBLinDynamics, SystemDynamics

class PolynomialSystem(SystemDynamics, FBLinDynamics):
    def __init__(self, root=1, drift_gain=1, act_gain=1):
        SystemDynamics.__init__(self, n=1, m=1)
        FBLinDynamics.__init__(self, relative_degrees=[1])
        self.root = root
        self.drift_gain = drift_gain
        self.act_gain = act_gain

    def drift(self, x, t):
        return -self.drift_gain * x * (x - self.root) * (x + self.root)

    def act(self, x, t):
        return array([[self.act_gain]])
