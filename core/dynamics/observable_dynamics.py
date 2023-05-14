from abc import ABC, abstractmethod
from torch.autograd.functional import jacobian
from system_dynamics import SystemDynamics
class ObservableDynamics(ABC):
    def __init__(self, obs_dim):
        self.obs_dim = obs_dim

    @abstractmethod
    def get_observation(self, state):
        pass

    @abstractmethod
    def to_principal_coordinates(self, state):
        pass

    @property
    def p(self):
        return self.obs_dim

    def make_obervable_dynamics(self):
        """ Returns the dynamics in the observable coordinates.

        :param observations: state in the original coordinates (B x n)
        :param actions: actions in the original coordinates (B x m)
        :param t: time
        :return: dynamics in the observable coordinates (B x obs_dim)
        """

        class ObservableSystem(SystemDynamics):
            def __init__(inner_self):
                SystemDynamics.__init__(inner_self, n=self.p, m=self.m)

            def forward(inner_self, y, u, t):
                x = self.to_principal_coordinates(y)
                dydx = self.jacobian(x)
                return dydx @ self(x, u, t)

        return ObservableSystem()

    def jacobian(self, observations):
        """ Returns the jacobian of the dynamics in the observable coordinates.

        :param observations: state in the original coordinates (B x n)
        :param actions: actions in the original coordinates (B x m)
        :param t: time
        :return: jacobian of the dynamics in the observable coordinates (B x obs_dim x n)
        """
        F = jacobian(lambda x: self.get_observation(x).sum(dim=0),
                 (observations),
                 create_graph=True)
        F = F.swapaxes(1,0)
        return F