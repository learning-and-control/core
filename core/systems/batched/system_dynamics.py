from abc import ABC, abstractmethod
from torch import stack, tensor
from torch.nn import Module
from torchdiffeq import odeint


class SystemDynamics(ABC, Module):
    @abstractmethod
    def forward(self, xs, us):
        raise NotImplementedError

    def step(self, x_0s, u_0s, t_0, t_f, atol=1e-6, rtol=1e-6):
        return odeint(lambda t, y: self(y, u_0s), x_0s, tensor([t_0, t_f]), atol=atol, rtol=rtol)[-1]

    def simulate(self, x_0s, controller, ts, atol=1e-6, rtol=1e-6):
        state_trajectories = [x_0s]
        action_trajectories = []
        for t_0, t_f in zip(ts[:-1], ts[1:]):
            x_0s = state_trajectories[-1]
            u_0s = controller(x_0s)
            x_fs = self.step(x_0s, u_0s, t_0, t_f, atol, rtol)
            state_trajectories.append(x_fs)
            action_trajectories.append(u_0s)

        state_trajectories = stack(state_trajectories)
        action_trajectories = stack(action_trajectories)
        return state_trajectories, action_trajectories
