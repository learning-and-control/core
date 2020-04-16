import pytest
import numpy as np
import torch as th
from core.systems import CartPole, InvertedPendulum
from scipy.linalg import expm
g = 9.18
m = 0.25
l = 0.5
T = 5
dt = 1e-2
f = 1 / dt
N = int(T * f)
ts = np.linspace(0, T, N + 1)

m_c = m * 0.5
m_p = m * 0.25
l_c = l * .5

th.manual_seed(0)


def analytic_cartpole_jacobian(x, u, t):
    # test numpy jacobian
    x, th, xd, thd = x

    div1 = (m_c + m_p - m_p * (np.cos(th) ** 2)) ** 2
    J21 = m_p * (
            g * m_p - g * (2 * m_c + m_p) * np.cos(2 * th) -
            2 * u * np.sin(2 * th) + l_c * np.cos(th) * (
                    2 * m_c - m_p + m_p * np.cos(2 * th)) * (thd ** 2)) / \
          (2 * div1)
    J23 = - (4 * l_c * m_p * np.sin(th) * thd) / (
            -2 * m_c - m_p + m_p * np.cos(2 * th))
    J24 = 1 / (m_c + m_p - m_p * np.cos(th) ** 2)
    J31 = (g * (m_c + m_p) * np.cos(th) * (
            2 * m_c - m_p + m_p * np.cos(2 * th)) +
           u * (2 * m_c + 3 * m_p + m_p * np.cos(2 * th)) * np.sin(th) +
           l_c * m_p * (m_p - (2 * m_c + m_p) * np.cos(2 * th)) * thd ** 2) / \
          (2 * l_c * div1)
    J33 = -(m_p * np.sin(2 * th) * thd) / (m_c + m_p - m_p * np.cos(th) ** 2)
    J34 = - np.cos(th) / (l_c * (m_c + m_p - m_p * np.cos(th) ** 2))
    return np.array([[0, 0, 1, 0, 0],
                     [0, 0, 0, 1, 0],
                     [0, J21, 0, J23, J24],
                     [0, J31, 0, J33, J34]])


def analytic_pendulum_jacobian(x, u, t):
    th, thd = x
    return np.array([[0, 1, 0],
                     [g * np.cos(th) / l, 0, 1 / (l ** 2 * m)]])


@pytest.mark.parametrize('system,analytic_jacobian',
                         [
                             (CartPole(m_c=m_c, m_p=m_p, l=l_c, g=g),
                              analytic_cartpole_jacobian),
                             (InvertedPendulum(m=m, l=l, g=g),
                              analytic_pendulum_jacobian)
                         ])
def test_system_jacboian(system, analytic_jacobian):
    n_tests = 3
    x_rands = th.rand((n_tests, system.n), dtype=th.float64)
    u_rands = th.rand((n_tests, system.m), dtype=th.float64)
    delta_t = 0.1234
    t = th.zeros((1))
    th_jacobians = []
    np_jacobians = []
    for i in range(n_tests):
        x = x_rands[i, :]
        u = u_rands[i, :]
        th_jacobian = system.jacobian(x, u, t)
        np_jacobian = system.jacobian(x.numpy(), u.numpy(), t.numpy())
        np_exp_jacobian = system.jacobian_exp(x.numpy(), u.numpy(), t.numpy(), delta_t)
        true_jacobian = analytic_jacobian(x.numpy(), u, t)
        true_A  = true_jacobian[:, :system.n]
        true_B = true_jacobian[:, system.n:]

        true_exp_jacobian = np.concatenate([
            expm(true_A * delta_t),
            np.linalg.pinv(true_A) @ (
                expm(true_A * delta_t) - np.eye(system.n)) @ true_B
        ], axis=1)

        assert len(th_jacobian) == len(np_jacobian) == 2

        np.testing.assert_array_almost_equal(true_exp_jacobian,
                                             np.concatenate(np_exp_jacobian,axis=1))
        np.testing.assert_array_almost_equal(true_jacobian,
                                             np.concatenate(np_jacobian,
                                                            axis=1))
        th.testing.assert_allclose(th.cat(th_jacobian, dim=1),
                                   th.tensor(true_jacobian))
