from matplotlib.pyplot import figure
from numpy import array
import numpy as np
from torch import cos, float64, sin, stack, tensor, atan2
from torch.nn import Module, Parameter
from core.dynamics import FullyActuatedRoboticDynamics, ObservableDynamics
from core.util import default_fig


class InvertedPendulum(FullyActuatedRoboticDynamics, ObservableDynamics):
    def __init__(self, mass, l, g=9.81):
        FullyActuatedRoboticDynamics.__init__(self, 1, 1)
        ObservableDynamics.__init__(self, 3)
        self.params = Parameter(tensor([mass, l, g], dtype=float64))

    def forward(self, x, u, t):
        theta_dot = x[..., 1]
        theta_ddot = (self.g/self.l)*sin(x[..., 0]) + u[..., 0]/(self.mass * self.l**2)
        xdot = stack([theta_dot, theta_ddot], dim=-1)
        return xdot

    def get_observation(self, state):
        theta = state[..., 0]
        theta_dot = state[..., 1]
        return stack([
            cos(theta),
            sin(theta),
            theta_dot
        ], dim=-1)

    def invert_observation(self, obs):
        x = obs[..., 0]
        y = obs[..., 1]
        dtheta = obs[..., 2]
        theta = atan2(y, x)
        return stack([theta, dtheta], dim=-1)

    def to_principal_coordinates(self, state):
        theta = state[..., 0]
        theta_dot = state[..., 1]
        principal_theta = atan2(sin(theta), cos(theta)) #y, x
        return stack([
            principal_theta,
            theta_dot
        ], dim=-1)

    def D(self, q):
        return (self.mass * (self.l ** 2))[None, None, None].expand(q.shape[0], -1, -1)

    def C(self, q, q_dot):
        return tensor([[[0.0]]], dtype=float64).expand(q.shape[0], -1, -1)

    def U(self, q):
        theta = q[:, 0, None]
        return self.mass * self.g * self.l * cos(theta)

    def G(self, q):
        theta = q[:, 0, None]
        return -self.mass * self.g * self.l * sin(theta)

    def B(self, q):
        return tensor([[[1]]], dtype=float64).expand(q.shape[0], -1, -1)
    
    @property
    def l(self):
        return self.params[1]

    @property
    def mass(self):
        return self.params[0]

    @property
    def g(self):
        return self.params[2]

    def plot_states(self, ts, xs, fig=None, ax=None, labels=None,
                    color="black"):
        fig, ax = default_fig(fig, ax)
        # angle = xs[:, 0]
        # angle = np.arctan2(np.sin(angle), np.cos(angle))
        # xs = np.stack([angle, xs[:, 1]], axis=1)
        ax.set_title('States', fontsize=16)
        ax.set_xlabel('$\\theta$ (rad)', fontsize=16)
        ax.set_ylabel('$\\dot{\\theta}$ (rad / sec)', fontsize=16)
        ax.set_xlim(-2 * np.pi, 2 * np.pi)
        ax.set_ylim(-10, 10)
        ax.scatter(*xs.T, linewidth=1, color=color)

        return fig, ax

    def plot_physical(self, ts, xs, fig=None, ax=None, skip=1):
        fig, ax = default_fig(fig, ax)

        _, l, g = self.params
        l = l.item()
        g = g.item()
        thetas = xs[:, 0]
        rs = l * array([np.sin(thetas), np.cos(thetas)])
        zs = 0 * thetas[::skip]

        ax.set_title('Physical space', fontsize=16)
        ax.set_xlabel('$x$ (m)', fontsize=16)
        ax.set_ylabel('$z$ (m)', fontsize=16)
        ax.plot([zs, rs[0, ::skip]], [zs, rs[1, ::skip]], 'k')
        ax.plot(*rs, linewidth=3)
        ax.set_xlim(-l - 0.2, l + 0.2)
        ax.set_ylim(-l - 0.2, l + 0.2)
        ax.axis('equal')

        return fig, ax

    def plot(self, xs, us, ts, fig=None, action_labels=None, skip=1):
        if fig is None:
            fig = figure(figsize=(12, 6), tight_layout=True)

        physical_ax = fig.add_subplot(1, 3, 1)
        fig, physical_ax = self.plot_physical(ts, xs, fig, physical_ax, skip)

        state_ax = fig.add_subplot(1, 3, 2)

        fig, state_ax = self.plot_states(ts, xs, fig, state_ax)

        action_ax = fig.add_subplot(1, 3, 3)
        fig, action_ax = self.plot_actions(ts, us, fig, action_ax,
                                           action_labels)

        return fig, (physical_ax, action_ax)
