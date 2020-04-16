import pytest
from numpy import array, concatenate, diag, eye, linspace, pi, ones, zeros
from numpy.linalg import norm
from numpy import testing
from core.systems import LinearSystemDynamics, InvertedPendulum
from core.controllers import ConstantController, QPController, \
    FBLinController, LQRController, PiecewiseConstantController
from core.trajopt import TrajectoryOptimizer
import matplotlib.pyplot as plt

linear_dyn = LinearSystemDynamics(
    array([
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [-1, 0, 0, 0],
        [0, 1, 0, 0],
    ]), array([
        [0, 0],
        [0, 0],
        [1, 0],
        [0, 1]
    ]))


def plot_2D_q_sys(xs, ax=None):
    if ax is None:
        fig = plt.figure(1)
        ax = fig.add_subplot()
    end_circ = plt.Circle((50, 50), 2, color='g')
    start_circ = plt.Circle((0, 0), 2, color='b')
    for x in xs:
        vnorm = norm(x[2:])
        vnorm = vnorm if vnorm > 0 else 1
        ax.arrow(x[0], x[1], x[2] / vnorm, x[3] / vnorm,
                 width=0.5,
                 length_includes_head=True)
    ax.set_xlim(-100, 100)
    ax.set_ylim(-100, 100)
    ax.add_artist(end_circ)
    ax.add_artist(start_circ)
    return ax


def test_reach_goal():
    ctrl = ConstantController(linear_dyn, array([0.0, 0.0]))
    xs, us = linear_dyn.simulate(x_0=[0, 0, 10, 0],
                                 controller=ctrl,
                                 ts=linspace(0, 100, 100))
    x_0 = array([0, 0, 10, 0])
    x_f = array([50, 50, 0, 0])
    trajopt = TrajectoryOptimizer(100, 0.1, linear_dyn)
    trajopt.add_hard_terminal_constraint(x_f)
    trajopt.add_static_quad_cost()
    [xt, ut] = trajopt.eval(x_0)
    ut.min()
    ut.max()
    testing.assert_allclose(xt[-1], x_f, atol=1e-7, rtol=1e-7)
    # uncomment for debugging
    # ax = plot_2D_q_sys(xt)
    # plt.show()

def test_continuous_approx():
    ctrl = ConstantController(linear_dyn, array([0.0, 0.0]))
    ts, hk = linspace(0, 100, 100,retstep=True)
    xs, us = linear_dyn.simulate(x_0=[0, 0, 10, 0],
                                 controller=ctrl,
                                 ts=ts)
    x_0 = array([0, 0, 10, 0])
    x_f = array([50, 50, 0, 0])
    trajopt = TrajectoryOptimizer(100, hk, linear_dyn, TrajectoryOptimizer.COLLOCATION_MODE.CTN_ONE_PT)
    trajopt.add_hard_terminal_constraint(x_f)
    trajopt.add_static_quad_cost()
    [xt, ut] = trajopt.eval(x_0)
    xs, us = linear_dyn.simulate(x_0=[0, 0, 10, 0],
                                 controller=PiecewiseConstantController(linear_dyn, hk, ut),
                                 ts=ts)
    testing.assert_allclose(xs, xt, atol=1e-1)
    testing.assert_allclose(xt[-1], x_f, atol=1e-7, rtol=1e-7)


def test_input_constraints():
    x_0 = array([0, 0, 10, 0])
    x_f = array([50, 50, 0, 0])
    trajopt = TrajectoryOptimizer(100, 0.1, linear_dyn)
    trajopt.add_static_quad_cost()
    trajopt.add_terminal_cost(Q_f=eye(4) * 1e5, offset=x_f)
    trajopt.add_input_constraints(u_min=ones(2) * -50, u_max=ones(2) * 1)
    [xt, ut] = trajopt.eval(x_0)

    if ut.min() < -50:
        testing.assert_allclose(ut.min(), -50, rtol=1e-4)
    if ut.max() <= 1:
        testing.assert_allclose(ut.max(), -50, rtol=1e-4)

    testing.assert_allclose(xt[-1], x_f, atol=1, rtol=1e-2)

    #uncomment for debugging
    #ax = plot_2D_q_sys(xt)
    #plt.show()

@pytest.mark.skip(
    reason="can be used for debugging but takes too long otherwise")
def test_nl_pendulum_with_constraints():
    from core.systems.inverted_pendulum import InvertedPendulum
    m = 0.25
    l = 0.5
    system = InvertedPendulum(m=m, l=l)
    T = 100
    ts, hk = linspace(0, 4, T,  retstep=True)
    x_0 = array([-pi, 0.0])
    # [xs, us] = system.simulate(x_0=x_0, controller=qp, ts=ts)
    def sqp_step(ws_ut):
        trajopt = TrajectoryOptimizer(T, hk, system,
                                      # max_delta_x=array([1 * pi / 180, 0.1]),
                                      max_delta_u=array([1]),
                                      # soft_dyn_weight=1e-3,
                                      collocation_mode=TrajectoryOptimizer.COLLOCATION_MODE.TRAPEZOIDAL)

        trajopt.add_static_quad_cost(Q=array([[1e3, 0],[0, 1e2]]))
        trajopt.add_terminal_cost(array([[1e7, 0],[0, 1e6]]))
        trajopt.add_input_constraints(u_min=ones(1)*-.3, u_max=ones(1)*.3)
        return trajopt.eval(x_0, max_cvx_iters=10, ws_ut=ws_ut)

    ut = 0.3 * zeros((T, 1))
    for i in range(0, T, 20):
        ut[i:i + 10] = -ut[i:i + 10]
    for i in range(50):
        [xt, ut] = sqp_step(ut)
        # system.plot(xt, ut[1:], ts)
        # plt.show()
    [xs, us] = system.simulate(x_0, PiecewiseConstantController(system, hk,ut), ts)

    # system.plot(xs, us, ts)

def test_double_inv_pendulum_trapezoiodal_vs_linearization():
    from core.systems.double_inverted_pendulum import DoubleInvertedPendulum
    system = DoubleInvertedPendulum(m_1=0.25, m_2=0.25,
                                    l_1=0.5, l_2=0.5)

    T = 100
    ts, hk = linspace(0, 4, T, retstep=True)
    x_0 = array([-pi, 0.0, 0.0, 0.0])

    def sqp_step(mode):
        trajopt = TrajectoryOptimizer(T, hk, system,
                                      # max_delta_x=array([1 * pi / 180, 0.1]),
                                      max_delta_u=array([0.5]),
                                      # soft_dyn_weight=1e-3,
                                      solver='GUROBI',
                                      collocation_mode=mode)

        # trajopt.build(x_0)
        # trajopt.add_static_quad_cost()
        trajopt.add_static_quad_cost(Q=diag(array([1e3, 1e3, 1e2, 1e2])))
        trajopt.add_terminal_cost(diag(array([1e7, 1e7, 1e6, 1e6])))
        trajopt.add_input_constraints(u_min=ones(2) * -.3, u_max=ones(2) * .3)
        return trajopt.eval(x_0, max_cvx_iters=10)

    [xt_trap, ut_trap] = sqp_step(TrajectoryOptimizer.COLLOCATION_MODE.TRAPEZOIDAL)
    [xs_trap, us_trap] = system.simulate(x_0, PiecewiseConstantController(system, hk, ut_trap), ts)
    [xt_lin, ut_lin] = sqp_step(TrajectoryOptimizer.COLLOCATION_MODE.CTN_ONE_PT)
    [xs_lin, us_lin] = system.simulate(x_0,PiecewiseConstantController(system,hk,ut_lin),ts)

    trap_err = xs_trap - xt_trap
    lin_err = xs_lin - xt_lin
    assert abs(trap_err).mean() > abs(lin_err).mean()
    # plt.semilogy(norm(trap_err,axis=1), 'r', label='Trapezoidal Collocation Error')
    # plt.semilogy(norm(lin_err, axis=1), 'b', label='Linearization Error')
    # plt.title('Double Pendulum Open Loop Error')
    # plt.legend(loc='upper left')
    # plt.show()

def  test_nl_constrained_pendulum_trapezoidal_vs_linearization():
    from core.systems.inverted_pendulum import InvertedPendulum
    m = 0.25
    l = 0.5
    system = InvertedPendulum(m=m, l=l)
    T = 100
    ts, hk = linspace(0, 4, T, retstep=True)
    x_0 = array([-pi, 0.0])

    def sqp_step(mode):
        trajopt = TrajectoryOptimizer(T, hk, system,
                                      # max_delta_x=array([1 * pi / 180, 0.1]),
                                      max_delta_u=array([1]),
                                      # soft_dyn_weight=1e-3,
                                      collocation_mode=mode)

        # trajopt.build(x_0)
        ws_ut = zeros((T,1))
        # trajopt.add_static_quad_cost()
        trajopt.add_static_quad_cost(Q=array([[1e3, 0], [0, 1e2]]))
        trajopt.add_terminal_cost(array([[1e7, 0], [0, 1e6]]))
        trajopt.add_input_constraints(u_min=ones(1) * -.3, u_max=ones(1) * .3)
        return trajopt.eval(x_0, max_cvx_iters=10, ws_ut=ws_ut)

    [xt_trap, ut_trap] = sqp_step(TrajectoryOptimizer.COLLOCATION_MODE.TRAPEZOIDAL)
    [xs_trap, us_trap] = system.simulate(x_0, PiecewiseConstantController(system, hk, ut_trap), ts)
    [xt_lin, ut_lin] = sqp_step(TrajectoryOptimizer.COLLOCATION_MODE.CTN_ONE_PT)
    [xs_lin, us_lin] = system.simulate(x_0,PiecewiseConstantController(system,hk,ut_lin),ts)

    trap_err = xs_trap - xt_trap
    lin_err = xs_lin - xt_lin
    assert abs(trap_err).mean() > abs(lin_err).mean()
    #uncomment plotting for debugging

    # plt.semilogy(norm(trap_err,axis=1), 'r', label='Trapezoidal Collocation Error')
    # plt.semilogy(norm(lin_err, axis=1), 'b', label='Linearization Error')
    # plt.title('Pendulum Open Loop Error')
    # plt.legend(loc='upper left')
    # plt.show()



def  test_linear_integration():
    from core.systems.inverted_pendulum import InvertedPendulum
    system = linear_dyn
    T = 100
    ts, hk = linspace(0, 10, T, retstep=True)
    x_0 = array([0, 0, 10, 0])
    x_f = array([50, 50, 0, 0])
    def sqp_step(mode):
        trajopt = TrajectoryOptimizer(T, hk, system,
                                      collocation_mode=mode)
        trajopt.add_static_quad_cost()
        trajopt.add_terminal_cost(Q_f=eye(4) * 1e5, offset=x_f)
        trajopt.add_input_constraints(u_min=ones(2) * -50, u_max=ones(2) * 1)
        return trajopt.eval(x_0=x_0, max_cvx_iters=1)

    [xt_lin, ut_lin] = sqp_step(TrajectoryOptimizer.COLLOCATION_MODE.CTN_ONE_PT)
    [xs_lin, us_lin] = system.simulate(x_0, PiecewiseConstantController(system,hk,ut_lin), ts)

    lin_err = xs_lin[:-1] - xt_lin[:-1]
    assert abs(lin_err.mean()) < 1e-4
    #uncomment plotting for debugging
    # fig, axs = plt.subplots(1,3)
    #
    # plot_2D_q_sys(xt_lin[:-2], axs[0])
    # plot_2D_q_sys(xs_lin[:-2], axs[1])
    # axs[2].semilogy(norm(lin_err, axis=1), 'b', label='Linearization Error')
    # axs[2].legend(loc='upper left')
    # plt.show()
