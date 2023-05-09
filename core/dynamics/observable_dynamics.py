from abc import ABC, abstractmethod


class ObservableDynamics(ABC):
    def __init__(self, obs_dim):
        self.obs_dim = obs_dim

    @abstractmethod
    def get_observation(self, state):
        pass

    @abstractmethod
    def to_principal_coordinates(self, state):
        pass