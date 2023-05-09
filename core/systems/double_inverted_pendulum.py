from matplotlib.pyplot import figure
from numpy import array
import numpy as np
from torch import cos, eye, float64, sin, stack, zeros, tensor, atan2, \
    unbind, ones_like
from torch.nn import Module, Parameter
from core.dynamics import FullyActuatedRoboticDynamics, ObservableDynamics
from core.util import default_fig

class DoubleInvertedPendulum(FullyActuatedRoboticDynamics, Module, ObservableDynamics):
    def __init__(self, m_1, m_2, l_1, l_2, g=9.81):
        FullyActuatedRoboticDynamics.__init__(self, 2, 2)
        ObservableDynamics.__init__(self, 6)
        self.params = Parameter(tensor([m_1, m_2, l_1, l_2, g], dtype=float64))

    def get_observation(self, state):
        theta1 = state[..., 0]
        theta2 = state[..., 1]
        theta1_dot = state[..., 2]
        theta2_dot = state[..., 3]
        return stack([
            cos(theta1),
            sin(theta1),
            cos(theta2),
            sin(theta2),
            theta1_dot,
            theta2_dot
        ], dim=-1)

    def to_principal_coordinates(self, state):
        theta1 = state[..., 0]
        theta2 = state[..., 1]
        theta1_dot = state[..., 2]
        theta2_dot = state[..., 3]
        p_theta1 = atan2(sin(theta1), cos(theta1)) #y, x
        p_theta2 = atan2(sin(theta2), cos(theta2)) #y, x
        return stack([
            p_theta1,
            p_theta2,
            theta1_dot,
            theta2_dot
        ], dim=-1)


    def D(self, q):
        m_1, m_2, l_1, l_2, _ = self.params
        _, theta_2 = unbind(q, dim=-1)
        D_11 = (m_1 + m_2) * (l_1 ** 2) + 2 * m_2 * l_1 * l_2 * cos(theta_2) + m_2 * (l_2 ** 2)
        D_12 = m_2 * l_1 * l_2 * cos(theta_2) + m_2 * (l_2 ** 2)
        D_21 = D_12
        D_22 = m_2 * (l_2 ** 2) * ones_like(D_11)
        return stack((stack([D_11, D_12], dim=-1),
                      stack([D_21, D_22], dim=-1)), dim=1)

    def C(self, q, q_dot):
        _, m_2, l_1, l_2, _ = self.params
        n_batch = q.shape[0]
        theta_2 = q[:, 1]
        theta_1_dot = q_dot[:, 0]
        theta_2_dot = q_dot[:, 1]
        C_11 = zeros((n_batch,), dtype=q.dtype, device=q.device)
        C_12 = -m_2 * l_1 * l_2 * (2 * theta_1_dot + theta_2_dot) * sin(theta_2)
        C_21 = -C_12 / 2
        C_22 = -m_2 * l_1 * l_2 * theta_1_dot * sin(theta_2) / 2
        return stack((stack([C_11, C_12], dim=-1),
                      stack([C_21, C_22], dim=-1)), dim=1)

    def B(self, q):
        return eye(2, dtype=q.dtype, device=q.device)[None].expand(q.shape[0], -1, -1)

    def U(self, q):
        m_1, m_2, l_1, l_2, g = self.params
        theta_1, theta_2 = q
        return (m_1 + m_2) * g * l_1 * cos(theta_1) + m_2 * g * l_2 * cos(theta_1 + theta_2)

    def G(self, q):
        m_1, m_2, l_1, l_2, g = self.params
        theta_1, theta_2 = unbind(q, dim=-1)
        G_1 = -(m_1 + m_2) * g * l_1 * sin(theta_1) - m_2 * g * l_2 * sin(theta_1 + theta_2)
        G_2 = -m_2 * g * l_2 * sin(theta_1 + theta_2)
        return stack([G_1, G_2], dim=-1)

    def forward(self, X, U, t):
        # unpacking params and states
        m1, m2, l1, l2, g = self.params
        th1 = X[..., 0] + np.pi
        th2 = X[..., 1]
        dth1 = X[..., 2]
        dth2 = X[..., 3]
        u1 = U[..., 0]
        u2 = U[..., 1]

        # optimized forward dynamics
        t2 = cos(th2)
        t3 = sin(th1)
        t4 = sin(th2)
        t5 = th1 + th2
        t6 = 2 * dth1
        t7 = l1 ** 2
        t8 = t2 ** 2
        t9 = l1 * t2
        t10 = sin(t5)
        t11 = 1.0 / t7
        t12 = dth2 + t6
        t13 = m1 * t7
        t14 = m2 * t7
        t15 = m2 * t8
        t16 = l2 + t9
        t17 = l2 * m2 * t10
        t21 = dth2 * l1 * l2 * m2 * t4 * t12
        t18 = -t15
        t19 = g * t17
        t20 = m1 + m2 + t18
        t22 = 1.0 / t20
        ddth1 = (t21 + u1 - g * (t17 + l1 * t3 * (m1 + m2))) / (t13 + t14 - t8 * t14) + (t16 * (t19 - u2 - (dth1 * dth2 * l1 * l2 * m2 * t4) / 2.0 + (dth1 * l1 * l2 * m2 * t4 * t12) / 2.0)) / (l2 * t13 + l2 * t14 - l2 * t8 * t14)
        ddth2 = -(t11 * t16 * t22 * (t21 + u1 - g * (t17 + l1 * m1 * t3 + l1 * m2 * t3))) / l2 - (1.0 / l2 ** 2 * t11 * t22 * (t19 - u2 + dth1 ** 2 * l1 * l2 * m2 * t4) * (t13 + t14 + l2 ** 2 * m2 + l2 * m2 * t9 * 2.0)) / m2

        return stack((dth1, dth2, ddth1, ddth2), dim=-1)

    def plot_coordinates(self, ts, qs, fig=None, ax=None, labels=None):
        fig, ax = default_fig(fig, ax)

        ax.set_title('Coordinates', fontsize=16)
        ax.set_xlabel('$\\theta_1$ (rad)', fontsize=16)
        ax.set_ylabel('$\\theta_2$ (rad)', fontsize=16)
        ax.plot(*qs.T, linewidth=3)

        return fig, ax

    def plot_states(self, ts, xs, fig=None, ax=None, labels=None):
        fig, ax = default_fig(fig, ax)

        ax.set_title('States', fontsize=16)
        ax.set_xlabel('$\\theta$ (rad)', fontsize=16)
        ax.set_ylabel('$\\dot{\\theta}$ (rad / sec)', fontsize=16)
        ax.plot(xs[:, 0], xs[:, 2], linewidth=3, label='$\\theta_1$')
        ax.plot(xs[:, 1], xs[:, 3], linewidth=3, label='$\\theta_2$')
        ax.legend(fontsize=16)

        return fig, ax

    def plot_actions(self, ts, us, fig=None, ax=None, labels=None):
        fig, ax = default_fig(fig, ax)

        if labels is None:
            labels = ['$\\tau_1$ (N $\\cdot m$)', '$\\tau_2$ (N $\\cdot$ m)']

        ax.set_title('Actions', fontsize=16)
        ax.set_xlabel(labels[0], fontsize=16)
        ax.set_ylabel(labels[1], fontsize=16)
        ax.plot(*us.T, linewidth=3)

        return fig, ax

    def plot_tangents(self, ts, xs, fig=None, ax=None, skip=1):
        fig, ax = default_fig(fig, ax)

        ax.set_title('Tangent Vectors', fontsize=16)
        ax.set_xlabel('$\\theta_1$ (rad)', fontsize=16)
        ax.set_ylabel('$\\theta_2$ (rad)', fontsize=16)
        ax.plot(*xs[:, :2].T, linewidth=3)
        ax.quiver(*xs[::skip, :2].T, *xs[::skip, 2:].T, angles='xy')

        return fig, ax

    def plot_physical(self, ts, xs, fig=None, ax=None, skip=1):
        fig, ax = default_fig(fig, ax)

        _, _, l_1, l_2, _ = self.params.detach().numpy()
        theta_1s, theta_2s = xs[:, :2].T
        r_1s = l_1 * array([np.sin(theta_1s),
                            np.cos(theta_1s)])
        r_2s = r_1s + l_2 * array([np.sin(theta_1s + theta_2s),
                                   np.cos(theta_1s +theta_2s)])
        zs = 0 * theta_1s[::skip]

        ax.set_title('Physical space', fontsize=16)
        ax.set_xlabel('$x$ (m)', fontsize=16)
        ax.set_ylabel('$z$ (m)', fontsize=16)
        ax.plot([zs, r_1s[0, ::skip]], [zs, r_1s[1, ::skip]], 'k')
        ax.plot([r_1s[0, ::skip], r_2s[0, ::skip]], [r_1s[1, ::skip], r_2s[1, ::skip]], 'k')
        ax.plot(*r_1s, linewidth=3)
        ax.plot(*r_2s, linewidth=3)
        ax.axis('equal')

        return fig, ax

    def plot(self, xs, us, ts, fig=None, action_labels=None, skip=1):
        if fig is None:
            fig = figure(figsize=(12, 6), tight_layout=True)

        physical_ax = fig.add_subplot(1, 2, 1)
        fig, physical_ax = self.plot_physical(ts, xs, fig, physical_ax, skip)

        action_ax = fig.add_subplot(1, 2, 2)
        fig, action_ax = self.plot_actions(ts, us, fig, action_ax, action_labels)

        return fig, (physical_ax, action_ax)
