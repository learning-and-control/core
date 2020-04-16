import pytest

from numpy import allclose, array, cos, identity, linspace, sin
from numpy.linalg import norm
from core.controllers import FBLinController, LQRController, QPController, LinearController
from core.systems import DoubleInvertedPendulum, InvertedPendulum
from core.geometry import Ball
from core.util import arr_map

@pytest.fixture
def pendulum_with_controller():
    m = 0.25
    l = 0.5
    g = 9.81

    T = 10
    dt = 1e-2
    f = 1 / dt
    N = int(T * f)
    ts = linspace(0, T, N + 1)

    system = InvertedPendulum(m, l, g)
    k_p = 1
    k_d = 2
    K = m * (l ** 2) * array([[g / l + k_p, k_d]])
    controller = LinearController(system, K)
    return system, controller, ts


def test_pendulum_linear_controller(pendulum_with_controller):
    (system, controller, ts) = pendulum_with_controller
    x_0 = array([1, 0])
    xs, _ = system.simulate(x_0, controller, ts)
    # Check convergence
    assert allclose(norm(xs, axis=1)[-10:], 0, atol=1e-3)



def test_pendulum_qp_controller(pendulum_with_controller):
    (system, controller, ts) = pendulum_with_controller

    ball = Ball(2)
    states = ball.sample(1000)

    invariant_controller = QPController(system, m=1)
    invariant_controller.add_regularizer(controller)
    invariant_controller.add_safety_constraint(ball.safety(system), lambda r: r)
    xs, _ = system.simulate(x_0=array([1, 0]),
                            controller=invariant_controller,
                            ts=ts)
    # Check convergence
    assert allclose(norm(xs, axis=1)[-10:], 0, atol=1e-3)


def test_double_pend_with_fb_linearize():
    T = 30
    dt = 1e-2
    f = 1 / dt
    N = int(T * f)
    ts = linspace(0, T, N + 1)

    m_1 = .25
    m_2 = .25
    l_1 = .5
    l_2 = .5
    g = 9.81
    system = DoubleInvertedPendulum(m_1, m_2, l_1, l_2, g)

    x_0 = array([1, 0, 0, 0])

    controller = FBLinController(system,
                                 LQRController.build(system,
                                                     Q=identity(4),
                                                     R=identity(2)))
    xs, _ = system.simulate(x_0, controller, ts)
    # Check convergence
    assert allclose(norm(xs, axis=1)[-10:], 0, atol=1e-3)
