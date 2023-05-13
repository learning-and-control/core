from abc import ABC, abstractmethod
from torch.autograd.functional import jacobian

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

    def obervable_dynamics(self, observations, actions, t):
        """ Returns the dynamics in the observable coordinates.

        :param observations: state in the original coordinates (B x n)
        :param actions: actions in the original coordinates (B x m)
        :param t: time
        :return: dynamics in the observable coordinates (B x obs_dim)
        """
        state = self.to_principal_coordinates(observations)
        return self(state, actions, t)

    def observable_dynamics_jacobian(self, observations, actions, t):
        """ Returns the jacobian of the dynamics in the observable coordinates.

        :param observations: state in the original coordinates (B x n)
        :param actions: actions in the original coordinates (B x m)
        :param t: time
        :return: jacobian of the dynamics in the observable coordinates (B x obs_dim x n)
        """
        F, G = jacobian(lambda y, u: self.observable_dynamics(y, u, t).sum(dim=0),
                 (observations, actions),
                 create_graph=True)
        F = F.swapaxes(1,0)
        G = G.swapaxes(1,0)
        return F, G