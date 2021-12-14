from matplotlib.pyplot import figure
from mpl_toolkits.mplot3d import Axes3D
from numpy import array
from torch import arange, atan, diag, float64, cat, cos, sin, \
    stack, tensor, zeros
from torch.nn import Module, Parameter

from core.dynamics import FBLinDynamics, RoboticDynamics, SystemDynamics
from core.util import default_fig


class PlanarQuadrotor(RoboticDynamics, Module):
    def __init__(self, m, J, g=9.81):
        RoboticDynamics.__init__(self, 3, 2)
        Module.__init__(self)
        self.params = Parameter(tensor([m, J, g]))

    def D(self, q):
        m, J, _ = self.params
        return diag(stack([m, m, J]))

    def C(self, q, q_dot):
        return tensor((3, 3), dtype=float64)

    def U(self, q):
        m, _, g = self.params
        _, z, _ = q
        return m * g * z

    def G(self, q):
        m, _, g = self.params
        _zero = tensor(0, dtype=float64)
        return stack([_zero, m * g, _zero])

    def B(self, q):
        _, _, theta = q
        _zero = tensor(0, dtype=float64)
        _one = tensor(1, dtype=float64)
        return stack((
            stack([sin(theta), _zero]),
            stack([cos(theta), _zero]),
            stack([_zero, _one])
        ))

    class Extension(SystemDynamics):
        def __init__(self, planar_quadrotor):
            SystemDynamics.__init__(self, n=8, m=2)
            self.quad = planar_quadrotor

        def step(self, x_0, u_0, t_0, t_f, atol=1e-6, rtol=1e-6):
            x = x_0[:6]
            f, f_dot = x_0[-2:]
            f_ddot, tau = u_0
            u = tensor([f, tau])

            dt = t_f - t_0
            f += f_dot * dt
            f_dot += f_ddot * dt
            x = self.quad.step(x, u, t_0, t_f, atol, rtol)

            return cat([x, cat([f, f_dot])])

    class Output(FBLinDynamics):
        def __init__(self, extension):
            relative_degrees = [4, 4]
            perm = cat([2 * arange(4), 2 * arange(4) + 1])
            FBLinDynamics.__init__(self, relative_degrees, perm)
            self.params = extension.quad.params

        def r_ddot(self, f, theta):
            m, _, g = self.params
            x_ddot = f * sin(theta) / m
            z_ddot = f * cos(theta) / m - g
            return cat((x_ddot, z_ddot))

        def r_dddot(self, f, f_dot, theta, theta_dot):
            m, _, _ = self.params
            x_dddot = (f_dot * sin(theta) + f * theta_dot * cos(theta)) / m
            z_dddot = (f_dot * cos(theta) - f * theta_dot * sin(theta)) / m
            return cat([x_dddot, z_dddot])

        def r_ddddot_drift(self, f, f_dot, theta, theta_dot):
            m, _, _ = self.params
            x_ddddot_drift = (2 * f_dot * theta_dot * cos(theta) - f * (
                    theta_dot ** 2) * sin(theta)) / m
            z_ddddot_drift = -(2 * f_dot * theta_dot * sin(theta) + f * (
                    theta_dot ** 2) * cos(theta)) / m
            return cat([x_ddddot_drift, z_ddddot_drift])

        def r_ddddot_act(self, f, theta):
            m, J, _ = self.params
            x_ddddot_act = tensor([sin(theta), f * cos(theta) / J]) / m
            z_ddddot_act = tensor([cos(theta), -f * sin(theta) / J]) / m
            return cat([x_ddddot_act, z_ddddot_act])

        def eval(self, x, t):
            q, q_dot = x[:6].view((2, 3))
            f, f_dot = x[-2:]
            r, theta = q[:2], q[-1]
            r_dot, theta_dot = q_dot[:2], q_dot[-1]
            r_ddot = self.r_ddot(f, theta)
            r_dddot = self.r_dddot(f, f_dot, theta, theta_dot)
            return cat([r, r_dot, r_ddot, r_dddot])

        def drift(self, x, t):
            eta = self.eval(x, t)
            theta, theta_dot, f, f_dot = x[tensor([2, 5, -2, -1])]
            r_ddddot_drift = self.r_ddddot_drift(f, f_dot, theta, theta_dot)
            return cat([eta[2:], r_ddddot_drift])

        def act(self, x, t):
            theta, f = x[2], x[-2]
            r_ddddot_act = self.r_ddddot_act(f, theta)
            return cat([zeros((6, 2)), r_ddddot_act])

        def to_state(self, eta):
            m, _, g = self.params

            r, r_dot = eta[:4].view((2, 2))
            x_ddot, z_ddot, x_dddot, z_dddot = eta[-4:]
            theta = atan(x_ddot / (z_ddot + g))
            theta_dot = ((z_ddot + g) * x_dddot - x_ddot * z_dddot) / (
                    (z_ddot + g) ** 2) * (cos(theta) ** 2)
            q = cat([r, theta])
            q_dot = cat([r_dot, theta_dot])

            f = m * (z_ddot + g) / cos(theta)
            f_dot = m * (z_dddot + x_ddot * theta_dot) / cos(theta)

            return cat([q, q_dot, cat([f, f_dot])])

    def plot_coordinates(self, ts, qs, fig=None, ax=None, labels=None):
        if fig is None:
            fig = figure(figsize=(6, 6), tight_layout=True)

        if ax is None:
            ax = fig.add_subplot(1, 1, 1, projection='3d')

        xs, zs, thetas = qs.T

        ax.set_title('Coordinates', fontsize=16)
        ax.set_xlabel('$x$ (m)', fontsize=16)
        ax.set_ylabel('$\\theta$ (rad)', fontsize=16)
        ax.set_zlabel('$z$ (m)', fontsize=16)
        ax.plot(xs, thetas, zs, linewidth=3)

        return fig, ax

    def plot_states(self, ts, xs, fig=None, ax=None, labels=None):
        fig, ax = default_fig(fig, ax)

        ax.set_title('States', fontsize=16)
        ax.set_xlabel('$q$', fontsize=16)
        ax.set_ylabel('$\\dot{q}$', fontsize=16)
        ax.plot(xs[:, 0], xs[:, 3], linewidth=3, label='$x$ (m)')
        ax.plot(xs[:, 1], xs[:, 4], linewidth=3, label='$z$ (m)')
        ax.plot(xs[:, 2], xs[:, 5], linewidth=3, label='$\\theta$ (rad)')
        ax.legend(fontsize=16)

        return fig, ax

    def plot_actions(self, ts, us, fig=None, ax=None, labels=None):
        fig, ax = default_fig(fig, ax)

        if labels is None:
            labels = ['$f$ (N)', '$\\tau$ (N $\\cdot$ m)']

        ax.set_title('Actions', fontsize=16)
        ax.set_xlabel(labels[0], fontsize=16)
        ax.set_ylabel(labels[1], fontsize=16)
        ax.plot(*us.T, linewidth=3)

        return fig, ax

    def plot_tangents(self, ts, xs, fig=None, ax=None, skip=1):
        fig, ax = default_fig(fig, ax)

        ax.set_title('Tangent Vectors', fontsize=16)
        ax.set_xlabel('$x$ (m)', fontsize=16)
        ax.set_ylabel('$z$ (m)', fontsize=16)
        ax.plot(*xs[:, :2].T, linewidth=3)
        ax.quiver(*xs[::skip, :2].T, *xs[::skip, 3:5].T, angles='xy')

        return fig, ax

    def plot_physical(self, ts, xs, fig=None, ax=None, skip=1):
        fig, ax = default_fig(fig, ax)

        xs, zs, thetas = xs[:, :3].T
        dirs = array([sin(thetas), cos(thetas)])[:, ::skip]

        ax.set_title('Physical Space', fontsize=16)
        ax.set_xlabel('$x$ (m)', fontsize=16)
        ax.set_ylabel('$z$ (m)', fontsize=16)
        ax.quiver(xs[::skip], zs[::skip], *dirs, angles='xy')
        ax.plot(xs, zs, linewidth=3)
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
