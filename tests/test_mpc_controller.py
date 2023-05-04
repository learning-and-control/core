import pytest
from numpy import array, concatenate, diag, eye, linspace, pi, ones, zeros, stack, zeros_like
from numpy.linalg import norm
from numpy import testing
from core.systems import LinearSystemDynamics, InvertedPendulum, \
    DoubleInvertedPendulum, Segway, SingleTrackBicycle, CartPole
from core.controllers import ConstantController, QPController, \
    FBLinController, LQRController, PiecewiseConstantController, MPCController
from core.trajopt import TrajectoryOptimizer
from os import makedirs
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import torch as th

root_save_dir = pathlib.Path(__file__).parent.absolute() / 'data' / 'test_gp_mpc'
th.set_grad_enabled(False)

def add_trajectory_arrows(ax, scale, traj, color='r'):
    for xt in traj:
        vnorm = norm(xt[2:]) * scale
        vnorm = vnorm if vnorm > 0 else 1
        ax.arrow(xt[0], xt[1], xt[2] / vnorm, xt[3] / vnorm,
                 width=0.5 / scale,
                 length_includes_head=False,
                 color=color)
def add_points(ax, coordinates, radius=1, color='r'):
    for g in coordinates:
        circle = plt.Circle((g[0], g[1]), radius, color=color, fill=True)
        ax.add_artist(circle)
def plot_2D_q_sys(xs, ax=None, traj=None,
                  xlim=(-100, 100), ylim=(-100, 100), scale=1.):
    if traj is not None:
        if isinstance(traj, list):
            for xt in traj:
                add_trajectory_arrows(ax, scale, xt)
        else:
            add_trajectory_arrows(ax, scale, traj)
    for x in xs:
        vnorm = norm(x[2:]) * scale
        vnorm = vnorm if vnorm > 0 else 1
        ax.arrow(x[0], x[1], x[2] / vnorm, x[3] / vnorm,

                 width=0.5/scale,
                 length_includes_head=False)
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    return ax

def plot_6D_config(xs, ax=None, traj=None,
                   start=None, end=None, xlim=None, ylim=None, scale=1):

    return plot_2D_q_sys(xs, ax, traj, start, end, xlim, ylim, scale)

def save_trajectory(us, xs, dir_name, suffix=''):
    save_dir = root_save_dir / dir_name
    makedirs(save_dir, exist_ok=True)
    np.save(str(save_dir / f'us{suffix}.npy'), us)
    np.save(str(save_dir / f'xs{suffix}.npy'), xs)

def test_mpc_linear_system():
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
    x_0 = array([0, 0, 10, 0])
    x_f = array([50, 50, 0, 0])
    xs = list()
    ts, hk = linspace(0, 2.5, 25, retstep=True)
    trajopt = TrajectoryOptimizer(T=10, h_k=hk,
                                  dynamics=linear_dyn,
                                  solver="GUROBI",
                                  collocation_mode=TrajectoryOptimizer.COLLOCATION_MODE.CTN_ONE_PT)
    trajopt.add_terminal_cost(Qf=eye(4) * 1e5, xf=x_f)
    trajopt.add_input_constraints(u_min=-70 * ones(2), u_max=50* ones(2))
    # trajopt.add_static_quad_cost()
    trajopt.add_static_quad_cost(Q=array([[1e2, 0, 0, 0],
                                         [0, 1e2, 0, 0],
                                         [0, 0, 1e1, 0],
                                         [0, 0, 0, 1e1]]), xf=x_f)
    def mpc_step(x, t, xt_prev, ut_prev):
        xs.append(x)
        xt, ut = trajopt.eval(x, max_cvx_iters=1, ws_xt=xt_prev, ws_ut=ut_prev)
        # ax = plot_2D_q_sys(xs, traj=xt)
        # plt.show()
        return xt, ut
    ctrl = MPCController(linear_dyn, mpc_step)
    xs, us = linear_dyn.simulate(x_0, controller=ctrl,
                                 ts=ts)
    # ctrl = ConstantController(linear_dyn, array([0.0, 0.0]))
    # testing.assert_allclose(xt[-1], x_f, atol=1e-7, rtol=1e-7)
    # uncomment for debugging
    testing.assert_allclose(x_f, xs[-1], atol=1e-1, rtol=1)
    # save_trajectory(us, xs, 'linear')
    # ax = plot_2D_q_sys(xs)
    # plt.show()


def test_mpc_pendulum():
    system = InvertedPendulum(mass=1.0, l=1.0)
    T = 350
    ts, hk = linspace(0, 3, T,  retstep=True)
    x_0s = th.tensor([[-pi, 0.0],
                      # [pi, 0.0],
                      # [0.0, .5],
                      # [0.0, -.5]
                     ])

    trajopt = TrajectoryOptimizer(T, hk, system,
                                  u_min=th.ones(1) * -2.0,
                                  u_max=th.ones(1) * 2.0,
                                  trapezoidal=False,
                                  # max_delta_x=array([1 * pi / 180, 0.1]),
                                  # max_delta_u=array([1])
                                  )
    Q_root = th.diag(th.sqrt(th.tensor([1e3, 1e2])))
    Qf_root = th.diag(th.sqrt(th.tensor([1e7, 1e6])))
    R_root = th.diag(th.sqrt(th.tensor([1e-3])))
    xf = th.tensor([
        [0.0, 0.0],
        # [np.pi, 0.0, 0.0, 0.0],
        # [np.pi/2, 0.0, 0.0, 0.0],
        # [-np.pi/2, 0.0, 0.0, 0.0]
    ])

    xs = list()
    def bmpc(x0, xf, xt_prev, ut_prev, max_cvx_iters=10, converge_tol=1e-3):
        return trajopt.eval(
            Q_sqrt=Q_root,
            R_sqrt=R_root,
            Qf_sqrt=Qf_root,
            x0=x0,
            xf=xf,
            ws_xt=xt_prev,
            ws_ut=ut_prev,
            max_cvx_iters=max_cvx_iters,
            converge_tol=converge_tol)
    def mpc_step(x, t, xt_prev, ut_prev):
        xs.append(x)

        xt, ut = bmpc(x, xf, xt_prev, ut_prev,
                      max_cvx_iters=10,
                      converge_tol=.01)
        # xt_prev = xt
        # ut_prev = ut
        if len(xs)%1 == 0:
            fig, ax = system.plot_states(ts,
                                         th.stack(xs,axis=0).flatten(0,1).detach().numpy(),
                                         color="black")
            radius = .5
            add_points(ax=ax, coordinates=xs[0].detach().numpy(),
                       radius=radius, color="blue")
            add_points(ax=ax, coordinates=xf.detach().numpy(),
                       radius=radius*.8, color="green")
            system.plot_states(ts, xt.flatten(0,1).detach().numpy(), color="red",
                               fig=fig,
                               ax=ax)
            plt.show()
        return xt, ut

    ctrl = MPCController(system, mpc_step)
    xs, us = system.simulate(x_0=x_0s,
                             ts=th.linspace(0,10,100),
                             controller=ctrl)

    testing.assert_allclose(actual=xs[:, -1], desired=zeros_like(xs[:, -1]),
                            atol=1e-1, rtol=1e-1)
    save_trajectory(us, xs, 'pendulum')
    system.plot_states(ts, xs)
    plt.show()

def test_mpc_cartpole():
    system = CartPole(m_c=1, m_p=0.1, l=0.5, g=9.8)
    T = 70
    ts, hk = linspace(0, 5, T,  retstep=True)
    ts = th.from_numpy(ts)
    x_0 = th.tensor([
        [0.0, 0.0, 0.0, 0.0],
        [3.0, 0.0, -1.0, 0.0],
        [3.0, 0.0, 1.0, 0.0],
        [-3.0, 0.0, 1.0, 0.0],
        [-3.0, 0.0, -1.0, 0.0],
        [0.0, -np.pi/2, 0.0, 0.0],
        [0.0, np.pi/2, 0.0, 1.0],
        [0.0, np.pi/2, 0.0, -1.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, -1.0],
    ])
    xf = th.tensor([
        [0.0, 0.0, 0.0, 0.0],
        # [np.pi, 0.0, 0.0, 0.0],
        # [np.pi/2, 0.0, 0.0, 0.0],
        # [-np.pi/2, 0.0, 0.0, 0.0]
    ])
    trajopt = TrajectoryOptimizer(T, hk, system,
                                  u_min=th.ones(1) * -10.,
                                  u_max=th.ones(1) * 10.,
                                  # max_delta_u=th.tensor([.01]),
                                  # max_delta_x=th.tensor([.01]),
                                    # max_delta_x=array([1 * pi / 180, 0.1]),
                                    # max_delta_u=th.tensor([1]),
    )

    R_root = th.diag(th.sqrt(th.tensor([1e-3])))
    Q_root = th.diag(th.sqrt(th.tensor([1e3, 1e3, 1e2, 1e2])))
    Qf_root = th.diag(th.sqrt(th.tensor([1e7, 1e7, 1e6, 1e6])))
    def bmpc(x0, xf, xt_prev, ut_prev, max_cvx_iters=10, converge_tol=1e-3):
        return trajopt.eval(
            Q_sqrt=Q_root,
            R_sqrt=R_root,
            Qf_sqrt=Qf_root,
            x0=x0,
            xf=xf,
            ws_xt=xt_prev,
            ws_ut=ut_prev,
            max_cvx_iters=max_cvx_iters,
            converge_tol=converge_tol)

    xs = list()
    def mpc_step(x, t, xt_prev, ut_prev):
        xs.append(x)
        xt, ut = bmpc(x, xf, xt_prev, ut_prev, max_cvx_iters=20, converge_tol=.01)
        # if len(xs)%5 == 0:
        #     fig, ax = plt.subplots(1)
        #     scale = 10
        #     radius = 0.2
        #     add_trajectory_arrows(
        #         ax=ax,
        #         scale=scale,
        #         traj=th.stack(xs, axis=0).flatten(0,1).detach().numpy(),
        #         color="black")
        #     add_trajectory_arrows(
        #         ax=ax,
        #         scale=scale,
        #         traj=xt.flatten(0,1).detach().numpy(),
        #         color="red")
        #     add_points(ax=ax, coordinates=xs[0].detach().numpy(),
        #                radius=radius, color="blue")
        #     add_points(ax=ax, coordinates=xf.detach().numpy(),
        #                radius=radius*.8, color="green")
        #     plt.show()
        return xt, ut

    controller = MPCController(system, mpc_step)
    (actual_xs, actual_us) = system.simulate(x_0=x_0,
                                             controller=controller,
                                             ts=ts)
    actual_xs = actual_xs.detach().numpy()
    actual_us = actual_us.detach().numpy()
    # for i in range(actual_xs.shape[0]):
    #     system.plot(actual_xs[i], actual_us[i], ts)
    #     plt.show()

    assert np.abs(actual_xs[:,-1]).max() < 1e-2

def double_pendulum_episode(x_0, x_f):
    dyn = DoubleInvertedPendulum(1.5, 1.5, 1, 1)
    ts, hk = linspace(0, 6, 100, retstep=True)
    trajopt = TrajectoryOptimizer(25, hk, dyn,
                                  max_delta_u=array([5, 5]),
                                  collocation_mode=TrajectoryOptimizer.COLLOCATION_MODE.CTN_ONE_PT,
                                  solver="GUROBI")
    trajopt.add_input_constraints(u_min=ones(2) * -8, u_max=ones(2) * 8)
    trajopt.add_static_quad_cost(Q=diag(array([1e3, 1e3, 1e1, 1e1])))
    trajopt.add_terminal_cost(diag(array([1e5, 1e5, 1e4, 1e4])))
    xs = list()
    xts = list()
    def mpc_step(x, t, xt_prev, ut_prev):
        xs.append(x)
        cvx_iter = 20 if xt_prev is None else 5
        xt, ut = trajopt.eval(x, max_cvx_iters=cvx_iter,
                              ws_xt=xt_prev, ws_ut=ut_prev,
                              converge_tol=1e-2)
        xts.append(xt[1:])
        # plot_2D_q_sys(xs, traj=xt,
        #               start=(x_0[0], x_0[1]),
        #               end=(x_f[0], x_f[1]),
        #               xlim=(-2*pi, 2*pi), ylim=(-2 * pi, 2 * pi),Angela
        #               scale=15)
        # plt.show()
        return xt, ut
    ctrl = MPCController(dyn, mpc_step)
    xs, us = dyn.simulate(x_0, controller=ctrl, ts=ts)
    plot_2D_q_sys(xs, traj=xts,
                  start=(x_0[0], x_0[1]),
                  end=(x_f[0], x_f[1]),
                  xlim=(-2*pi, 2*pi), ylim=(-2 * pi, 2 * pi),
                  scale=15)
    plt.show()
    # testing.assert_allclose(actual=xs[-1], desired=x_f, atol=1e-4, rtol=1e-4)
    return us, xs

@pytest.mark.skip(reason="Too Unstable Currently")
def test_mpc_double_pendulum():
    x_0 = array([-pi, 0, 0, 0])
    x_f = array([0, 0, 0, 0])
    from numpy import random
    noises = random.uniform(low=(-.5, -.5, -.2, -.2),
                   high=(.5, .5, .2, .2),
                   size=(1,4)
                   # size=(10,4)
                            )
    x_0s = x_0 + noises * 0
    for i in range(x_0s.shape[0]):
        us, xs = double_pendulum_episode(x_0s[i], x_f)
        # plot_2D_q_sys(xs, start=(x_0s[i][0], x_0s[i][1]),
        #               end=(x_f[0], x_f[1]),
        #               xlim=(-2*pi, 2*pi), ylim=(-2 * pi, 2 * pi),
        #               scale=15)
        # plt.show()
        # save_trajectory(us, xs, 'double_pendulum', suffix=i)


def test_mpc_segway():
    dyn = Segway()
    x_0 = array([-1, 0, 0, 0])
    x_f = array([1, 0, 0, 0])
    ts, hk = linspace(0, 6, 100, retstep=True)

    trajopt = TrajectoryOptimizer(30, hk, dyn,
                                  max_delta_u=array([5,5]),
                                  collocation_mode=TrajectoryOptimizer.COLLOCATION_MODE.CTN_ONE_PT,
                                  solver="GUROBI")
    trajopt.add_input_constraints(u_min=ones(1) * -15, u_max=ones(1) * 15)
    trajopt.add_static_quad_cost(Q=diag(array([1e3, 1e2, 1e1, 1e1])),
                                 xf=x_f)
    trajopt.add_terminal_cost(diag(array([1e5, 1e4, 1e3, 1e3])), xf=x_f)
    xs = list()
    def mpc_step(x, t, xt_prev, ut_prev):
        xs.append(x)
        cvx_iter = 20 if xt_prev is None else 5
        xt, ut = trajopt.eval(x, max_cvx_iters=cvx_iter,
                              ws_xt=xt_prev, ws_ut=ut_prev,
                              converge_tol=1e-2)
        # plot_2D_q_sys(xs, traj=xt, start=(-1, 0), end=(1, 0),
        #               xlim=(-2, 2), ylim=(- pi, pi),
        #               scale=20)
        # plt.show()
        return xt, ut
    ctrl = MPCController(dyn, mpc_step)
    xs, us = dyn.simulate(x_0, controller=ctrl, ts=ts)
    testing.assert_allclose(actual=xs[-1], desired=x_f, atol=1e-3, rtol=1e-3)
    save_trajectory(us, xs, 'segway')
    # plot_2D_q_sys(xs, start=(-1, 0), end=(1, 0),
    #               xlim=(-2, 2), ylim=(- pi, pi),
    #               scale=20)
    # plt.show()

def test_single_track_bicycle():
    dyn = SingleTrackBicycle()
    x_0 = array([-1, 0, 0, 0.1, 0, 0])
    x_f = array([1, 1, 0, 0.2, 0, 0])
    ts, hk = linspace(0, 6, 100, retstep=True)

    trajopt = TrajectoryOptimizer(30, hk, dyn,
                                  # max_delta_u=array([5,5]),
                                  collocation_mode=TrajectoryOptimizer.COLLOCATION_MODE.CTN_ONE_PT,
                                  solver="GUROBI")
    trajopt.add_input_constraints(u_min=ones(2) * -15, u_max=ones(2) * 15)
    trajopt.add_static_quad_cost(
        Q=diag(array([1e3, 1e3, 1e3, 1e1, 1e1, 1e1])),
        xf=x_f)
    trajopt.add_terminal_cost(diag(array([1e5, 1e5, 1e5, 1e2, 1e2, 1e2])), xf=x_f)
    xs = list()
    def mpc_step(x, t, xt_prev, ut_prev):
        xs.append(x)
        cvx_iter = 20 if xt_prev is None else 5
        xt, ut = trajopt.eval(x, max_cvx_iters=cvx_iter,
                              ws_xt=xt_prev, ws_ut=ut_prev,
                              converge_tol=1e-2)
        ax = plot_6D_config(xs, xt, start=(-1, 0), end=(1,1),
                          xlim=(-2, 2), ylim=(- pi, pi),
                          scale=20)
        plt.show()
        # plot_2D_q_sys(xs, traj=xt, start=(-1, 0), end=(1, 0),
        #               xlim=(-2, 2), ylim=(- pi, pi),
        #               scale=20)
        # plt.show()
        return xt, ut
    ctrl = MPCController(dyn, mpc_step)
    xs, us = dyn.simulate(x_0, controller=ctrl, ts=ts)
    testing.assert_allclose(actual=xs[-1], desired=x_f, atol=1e-3, rtol=1e-3)
    # save_trajectory(us, xs, 'single_track_bicycle')
    # plot_2D_q_sys(xs, start=(-1, 0), end=(1, 0),
    #               xlim=(-2, 2), ylim=(- pi, pi),
    #               scale=20)
    # plt.show()
