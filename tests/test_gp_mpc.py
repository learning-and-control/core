import pytest
from numpy import array, concatenate, diag, eye, linspace, pi, ones, zeros, \
    stack, zeros_like
from numpy.linalg import norm
from numpy import testing
from core.systems import LinearSystemDynamics, InvertedPendulum, \
    DoubleInvertedPendulum, Segway, AffineGPSystem
from core.controllers import ConstantController, QPController, \
    FBLinController, LQRController, PiecewiseConstantController, MPCController
from core.learning import GaussianProcess, RBFKernel, ScaledGaussianProcess, \
    GPScaler,AdditiveKernel, MultiplicativeKernel, PeriodicKernel
from core.trajopt import TrajectoryOptimizer, GPTrajectoryOptimizer
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import torch as th

root_save_dir = pathlib.Path(__file__).parent.absolute() / 'data' / 'test_gp_mpc'

def add_trajectory_arrows(ax, scale, traj):
    for xt in traj:
        vnorm = norm(xt[2:]) * scale
        vnorm = vnorm if vnorm > 0 else 1
        ax.arrow(xt[0], xt[1], xt[2] / vnorm, xt[3] / vnorm,
                 width=0.5 / scale,
                 length_includes_head=False,
                 color='r')

def plot_2D_q_sys(xs, ax=None, traj=None, start=(0,0), end=(50,50),
                  xlim=(-100, 100), ylim=(-100, 100), scale=1.):
    if ax is None:
        fig = plt.figure(1)
        ax = fig.add_subplot()
    end_circ = plt.Circle(end, 2/scale, color='g')
    start_circ = plt.Circle(start, 2/scale, color='b')
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
    ax.add_artist(end_circ)
    ax.add_artist(start_circ)
    return ax

def load_data(dir_name, suffix=''):
    us = np.load(root_save_dir / dir_name / f'us{suffix}.npy')
    xs = np.load(root_save_dir / dir_name / f'xs{suffix}.npy')
    X = concatenate([xs[:-1], us], axis=1)
    Y = xs[1:]
    return th.from_numpy(X), th.from_numpy(Y)

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
    n = 4
    m = 2
    xs = list()

    ts, hk = linspace(0, 2.5, 25, retstep=True)

    x_scaler = GPScaler(xmins=th.tensor([-100, -100, -100, -100, -100, -100]),
                        xmaxs=th.tensor([100, 100, 100, 100, 100, 100]))
    y_scaler = GPScaler(xmins=th.tensor([-100, -100, -100, -100]),
                        xmaxs=th.tensor([100, 100, 100, 100]))
    X, Y = load_data('linear')
    gp_est = ScaledGaussianProcess(X, Y, RBFKernel(n+m, ard_num_dims=True),
                                   x_scaler=x_scaler, y_scaler=y_scaler)
    gp_est.train_model(n_iters=70, lr=0.6)
    gp_dyn = AffineGPSystem(gp_est, n, m, hk)

    trajopt = GPTrajectoryOptimizer(T=10, h_k=hk,
                                  dynamics=gp_dyn,
                                  solver="GUROBI",
                                  # collocation_mode=TrajectoryOptimizer.COLLOCATION_MODE.CTN_ONE_PT
                                  )
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
        ax = plot_2D_q_sys(xs, traj=xt)
        plt.show()
        return xt, ut
    ctrl = MPCController(gp_dyn, mpc_step)
    xs, us = linear_dyn.simulate(x_0, controller=ctrl, ts=ts)
    # ctrl = ConstantController(linear_dyn, array([0.0, 0.0]))
    # testing.assert_allclose(xt[-1], x_f, atol=1e-7, rtol=1e-7)
    # uncomment for debugging
    testing.assert_allclose(x_f, xs[-1], atol=1e-1, rtol=1)
    ax = plot_2D_q_sys(xs)
    plt.show()

def test_mpc_pendulum():
    m = 0.25
    l = 0.5
    dyn = InvertedPendulum(mass=m, l=l)
    T = 35
    x_0 = array([-pi, 0.0])
    ts, hk = linspace(0, 3, T,  retstep=True)
    n = 2
    m = 1

    ts, hk = linspace(0, 2.5, 25, retstep=True)

    x_scaler = GPScaler(xmins=th.tensor([-2 *  pi, -pi, -.6]),
                        xmaxs=th.tensor([2 * pi, pi, .6]))
    y_scaler = GPScaler(xmins=th.tensor([-2 *  pi, -pi]),
                        xmaxs=th.tensor([2 * pi, pi]))
    X, Y = load_data('pendulum')
    kernel = AdditiveKernel(kernels=[
        MultiplicativeKernel(kernels=[PeriodicKernel(p_prior=2.,
                                                     learn_period=False),
                                       RBFKernel(1)],
                             active_dims=[[0], [0]]),
                              RBFKernel(2, ard_num_dims=True)],
                            active_dims=[[0], [1,2]])
    # kernel = RBFKernel(n + m, ard_num_dims=True)
    gp_est = ScaledGaussianProcess(X, Y, kernel,
                                   x_scaler=x_scaler, y_scaler=y_scaler)
    gp_est.train_model(n_iters=50, lr=0.5)
    gp_dyn = AffineGPSystem(gp_est, n, m, hk)


    trajopt = GPTrajectoryOptimizer(15, hk, gp_dyn,
                                  solver="GUROBI",
                                  # max_delta_x=array([1 * pi / 180, 0.1]),
                                  max_delta_u=array([1]),
                                  # soft_dyn_weight=1e-3,
                                  # collocation_mode=TrajectoryOptimizer.COLLOCATION_MODE.CTN_ONE_PT
                                  )

    trajopt.add_static_quad_cost(Q=array([[1e3, 0], [0, 1e2]]))
    trajopt.add_terminal_cost(array([[1e7, 0], [0, 1e6]]))
    trajopt.add_input_constraints(u_min=ones(1) * -.6, u_max=ones(1) * .6)
    xs = list()
    def mpc_step(x, t, xt_prev, ut_prev):
        xs.append(x)
        xt, ut = trajopt.eval(x, max_cvx_iters=10,
                              ws_xt=xt_prev, ws_ut=ut_prev,
                              converge_tol=1)
        fig, ax = dyn.plot_states(ts, stack(xs,axis=0), color="black")
        dyn.plot_states(ts, xt, color="red", fig=fig, ax=ax)
        plt.show()
        return xt, ut

    ctrl = MPCController(gp_dyn, mpc_step)
    xs, us = dyn.simulate(x_0=x_0, ts=ts, controller=ctrl)
    testing.assert_allclose(actual=xs[-1], desired=zeros_like(xs[-1]),
                            atol=1e-1, rtol=1e-1)
    dyn.plot_states(ts, xs)
    plt.show()

def test_mpc_double_pendulum():
    dyn = DoubleInvertedPendulum(1.5, 1.5, 1, 1)
    x_0 = array([-pi, 0, 0, 0])
    x_f = array([0, 0, 0, 0])
    ts, hk = linspace(0, 6, 100, retstep=True)
    n=4
    m=2
    X = list()
    Y = list()
    for i in range(5):
        X_i, Y_i = load_data('double_pendulum', suffix=i)
        X.append(X_i)
        Y.append(Y_i)
    X = th.cat(X, dim=0)
    Y = th.cat(Y, dim=0)
    x_scaler = GPScaler(xmins=th.tensor([-2 *  pi, -2 *  pi, -pi, -pi, -8,  -8]),
                        xmaxs=th.tensor([2 * pi, 2 * pi,  pi, pi, 8, 8]))
    y_scaler = GPScaler(xmins=th.tensor([-2 *  pi, -2 *  pi, -pi, -pi]),
                        xmaxs=th.tensor([2 * pi, 2 * pi,  pi, pi]))
    # kernel = AdditiveKernel(kernels=[
    #     MultiplicativeKernel(kernels=[PeriodicKernel(p_prior=2.,
    #                                                  learn_period=False),RBFKernel(1)], active_dims=[[0], [0]]),
    #     MultiplicativeKernel(kernels=[PeriodicKernel(p_prior=2.,
    #                                                  learn_period=False),
    #                                   RBFKernel(1)], active_dims=[[0], [0]]),
    #                           RBFKernel(4, ard_num_dims=True)],
    #                         active_dims=[[0], [1], [2, 3, 4, 5]])
    # kernel = RBFKernel(n + m, ard_num_dims=True)
    kernel = MultiplicativeKernel(kernels=[
        PeriodicKernel(p_prior=2.,learn_period=False),
        RBFKernel(1),
        PeriodicKernel(p_prior=2., learn_period=False),
        RBFKernel(1),
        RBFKernel(4, ard_num_dims=True),
    ],
                                  active_dims=[[0], [0], [1], [1], [2, 3, 4, 5]])
    gp_est = ScaledGaussianProcess(X, Y, kernel,
                                   x_scaler=x_scaler, y_scaler=y_scaler)
    # gp_est.train_model(n_iters=10, lr=0.2)
    #Periodic Kernel
    # gp_est.train_model(n_iters=210, lr=0.2)
    #RBF Kernel
    gp_est.train_model(n_iters=100, lr=0.2)
    gp_dyn = AffineGPSystem(gp_est, n, m, hk)

    trajopt = GPTrajectoryOptimizer(25, hk, gp_dyn,
                                  max_delta_u=array([5, 5]),
                                  # collocation_mode=TrajectoryOptimizer.COLLOCATION_MODE.CTN_ONE_PT,
                                  solver="GUROBI")
    trajopt.add_input_constraints(u_min=ones(2) * -8, u_max=ones(2) * 8)
    trajopt.add_static_quad_cost(Q=diag(array([1e3, 1e3, 1e1, 1e1])))
    trajopt.add_terminal_cost(diag(array([1e5, 1e5, 1e4, 1e4])))
    xs = list()
    xts = list()
    us = list()
    def mpc_step(x, t, xt_prev, ut_prev):

        cvx_iter = 20 if xt_prev is None else 5
        xt, ut = trajopt.eval(x, max_cvx_iters=cvx_iter,
                              ws_xt=xt_prev, ws_ut=ut_prev,
                              converge_tol=1e-2)

        if len(xs) > 0:
            xs_prev = xs[-1]
            us_prev = us[-1]
            sample = th.from_numpy(concatenate([xs_prev, us_prev])[None, :])
            xs_expected, cov = gp_est(sample)
            xs_expected = xs_expected.squeeze().detach().numpy()
            cov = cov.squeeze().detach().numpy()
            if (abs(xs_expected - x) / cov).max() > 0.25 or cov.max() > 0.1:
                print("[INFO] Found Surprising Example. Appending.")
                gp_est.add_samples(sample, th.from_numpy(x[None,:]))
                gp_est.train_model(1, lr=0.1)
        # plot_2D_q_sys(xs, traj=xt, end=(-pi, 0),
        #               xlim=(-2*pi, 2*pi), ylim=(-2 * pi, 2 * pi),
        #               scale=15)
        us.append(ut[0])
        xs.append(x)
        xts.append(xt[1:])
        # plt.show()
        # plt.pause(0.2)
        return xt, ut
    ctrl = MPCController(gp_dyn, mpc_step)
    xs, us = dyn.simulate(x_0, controller=ctrl, ts=ts)
    plot_2D_q_sys(xs, traj=xts, end=(-pi, 0),
                  xlim=(-2*pi, 2*pi), ylim=(-2 * pi, 2 * pi),
                  scale=15)
    plt.show()
    testing.assert_allclose(actual=xs[-1], desired=x_f, atol=1e-4, rtol=1e-4)

def test_mpc_segway():
    dyn = Segway()
    x_0 = array([-1, 0, 0, 0])
    x_f = array([1, 0, 0, 0])
    ts, hk = linspace(0, 6, 100, retstep=True)

    n=4
    m=1
    X, Y = load_data('segway')
    x_scaler = GPScaler(xmins=th.tensor([-2, -pi, -pi, -pi, -15 ]),
                        xmaxs=th.tensor([2, pi,  pi, pi, 15]))
    y_scaler = GPScaler(xmins=th.tensor([-2, -pi, -pi, -pi]),
                        xmaxs=th.tensor([2, pi,  pi, pi]))
    kernel = RBFKernel(n + m, ard_num_dims=True)
    gp_est = ScaledGaussianProcess(X, Y, kernel,
                                   x_scaler=x_scaler, y_scaler=y_scaler)
    gp_est.train_model(n_iters=50, lr=0.5)
    gp_dyn = AffineGPSystem(gp_est, n, m, hk)

    trajopt = GPTrajectoryOptimizer(30, hk, gp_dyn,
                                  max_delta_u=array([5,5]),
                                  # collocation_mode=TrajectoryOptimizer.COLLOCATION_MODE.CTN_ONE_PT,
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
    ctrl = MPCController(gp_dyn, mpc_step)
    xs, us = dyn.simulate(x_0, controller=ctrl, ts=ts)
    plot_2D_q_sys(xs, start=(-1, 0), end=(1, 0),
                  xlim=(-2, 2), ylim=(- pi, pi),
                  scale=20)
    plt.show()
    testing.assert_allclose(actual=xs[-1], desired=x_f, atol=1e-3, rtol=1e-3)

