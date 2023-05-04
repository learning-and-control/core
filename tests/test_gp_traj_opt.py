from numpy import concatenate, eye, linspace, pi, ones, random, \
    sin, cos, newaxis, stack, meshgrid, diag, diagonal
from numpy.core._multiarray_umath import array, zeros
from scipy.linalg import expm, logm
from numpy.linalg import norm, pinv
from numpy.testing import assert_allclose
from core.learning import GaussianProcess, RBFKernel, ScaledGaussianProcess, GPScaler
from core.systems import LinearSystemDynamics, InvertedPendulum, \
    AffineGPSystem, DoubleInvertedPendulum, CartPole
from core.controllers import ConstantController, PiecewiseConstantController
from core.trajopt import TrajectoryOptimizer, GPTrajectoryOptimizer

import matplotlib.pyplot as plt
import torch as th
import seaborn as sns

def plot_2D_q_sys(xs, ax=None, color='black', box_side=60,
                  start=(0,0), end=(50,50)):
    if ax is None:
        fig = plt.figure(1)
        ax = fig.add_subplot()
    end_circ = plt.Circle(end, .2, color='g')
    start_circ = plt.Circle(start, .2, color='b')
    for x in xs:
        vnorm = norm(x[2:]) * 10
        vnorm = vnorm if vnorm > 0 else 1
        ax.arrow(x[0], x[1], x[2] / vnorm, x[3] / vnorm,
                 width=0.05,
                 length_includes_head=True,
                 color=color)
    ax.set_xlim(-box_side, box_side)
    ax.set_ylim(-box_side, box_side)
    ax.add_artist(end_circ)
    ax.add_artist(start_circ)
    return ax

def plot_2D_dyn_sys(dyn,  ax=None,
                    low_x=-5, high_x=5,
                    low_y=-5, high_y=5,
                    n_sample=100):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot()
    X, Y = meshgrid(linspace(low_x, high_x, n_sample),
                    linspace(low_y, high_y, n_sample))
    arrows = array([dyn(stack(xy), zeros((1,)), 0) for xy in
                    zip(X.ravel(), Y.ravel())])
    arrows = arrows.reshape(X.shape + (2,))
    ax.streamplot(X, Y, arrows[:, :, 0], arrows[:, :, 1])
    # ax.quiver(X, Y, arrows[:, :, 0], arrows[:, :, 1])
    return arrows, ax

def test_2D_lin_learned_phase():
    true_A = array([
        [0, -3],
        [1, 0]
    ])
    true_B = array([
        [0],
        [1]])

    u_0 = array([0.0])

    dyn = LinearSystemDynamics(true_A, true_B)
    train_x0s = (random.rand(100, 2) - 0.5) * 10

    dt = 0.01
    train_xtp1s = [dyn.step(x_0=x0, u_0=u_0, t_0=0, t_f=dt) for x0 in train_x0s]
    train_xtp1s = stack(train_xtp1s, axis=0)
    train_X = concatenate((train_x0s, zeros((train_x0s.shape[0], 1))), axis=1)
    train_X = th.from_numpy(train_X)
    train_Y =  th.from_numpy(train_xtp1s)

    gp_est = GaussianProcess(train_X, train_Y, RBFKernel(2+1))
    gp_est.train_model()
    gp_dyn = AffineGPSystem(gp_est, n=2, m=1, delta_t=dt)
    fig, axs = plt.subplots(1,3)
    expected, _ = plot_2D_dyn_sys(dyn, axs[0])
    axs[0].set_title('Expected Phase Plot')
    actual, _ = plot_2D_dyn_sys(gp_dyn, axs[1])
    axs[1].set_title('Actual Phase Plot')
    error = norm(actual - expected,2, axis=2)
    assert error.mean() <= 1e-1
    #uncomment plotting for debugging
    sns.heatmap(error, ax=axs[2])
    axs[2].set_title('Error of Phase Plot')
    plt.show()


def test_double_pendulum_trajopt():
    T = 1
    K = 100
    true_dyn = DoubleInvertedPendulum(1.5,1.5, 1, 1)
    n = true_dyn.n
    m = true_dyn.m
    ts, hk = linspace(0, T, K, retstep=True)
    x_0 = array([-pi, 0, 0, 0])
    x_f = array([0, 0, 0, 0])

    n_data = 1000

    theta0s = random.uniform(-2*pi, 2*pi, (n_data,))
    theta1s = random.uniform(-2*pi, 2*pi, (n_data,))

    theta0_dots = random.uniform(-5, 5, (n_data,))
    theta1_dots =random.uniform(-5, 5, (n_data,))

    u0s = random.uniform(-2, 2, (n_data, 2))
    train_x0s = stack([theta0s, theta1s, theta0_dots, theta1_dots], axis=1)
    train_xtp1s = stack([
        true_dyn.step(x_0=x0, u_0=u0, t_0=0, t_f=hk)
        for (x0, u0) in zip(train_x0s, u0s)
    ], axis=0)
    X = concatenate([train_x0s, u0s], axis=1)
    X = th.from_numpy(X)
    Y = th.from_numpy(train_xtp1s)
    Y.max(dim=0)
    Y.min(dim=0)
    x_scaler = GPScaler(xmins=X.min(dim=0)[0],
                        xmaxs=X.max(dim=0)[0])
    y_scaler = GPScaler(xmins=X[:,:n].min(dim=0)[0],
                        xmaxs=X[:,:n].max(dim=0)[0])
    gp_est = ScaledGaussianProcess(X, Y, RBFKernel(n+m, ard_num_dims=True),
                                   x_scaler=x_scaler, y_scaler=y_scaler)
    gp_est.train_model(n_iters=200, lr=0.5)
    gp_dyn = AffineGPSystem(gp_est, n=n, m=m, delta_t=hk)

    gp_trajopt = GPTrajectoryOptimizer(K, hk, gp_dyn,
                                  # max_delta_x=array([pi/10, pi/10, 5/10, 5/10]),
                                  max_delta_u=array([0.05, 0.05]),
                                  solver='GUROBI')  # GUROBI because OSQP can
    # fail
    # gp_trajopt.add_state_box_constraints(
    #     x_min=X[:,:n].min(dim=0)[0].detach().numpy(),
    #     x_max=X[:,:n].max(dim=0)[0].detach().numpy())
    gp_trajopt.add_static_quad_cost(Q=diag(array([1e3, 1e3, 1e2, 1e2])))
    gp_trajopt.add_terminal_cost(diag(array([1e7, 1e7, 1e6, 1e6])))
    gp_trajopt.add_input_constraints(u_min=ones(2) * -2, u_max=ones(2) * 2)
    xt_gp, ut_gp = gp_trajopt.eval(x_0, max_cvx_iters=20)
    xs_gp, us_gp = true_dyn.simulate(x_0,
                               PiecewiseConstantController(gp_dyn, hk, ut_gp),
                               ts)
    xt_gp_im, ut_gp_im = gp_dyn.simulate(x_0,
                               PiecewiseConstantController(gp_dyn, hk, ut_gp),
                               ts)


    trajopt = TrajectoryOptimizer(K, hk, true_dyn,
                                  max_delta_u=array([0.5, 0.5]),
                                  collocation_mode=TrajectoryOptimizer.COLLOCATION_MODE.CTN_ONE_PT,
                                  solver='GUROBI')  # GUROBI because OSQP can fail
    trajopt.add_static_quad_cost(Q=diag(array([1e3, 1e3, 1e2, 1e2])))
    trajopt.add_terminal_cost(diag(array([1e7, 1e7, 1e6, 1e6])))
    trajopt.add_input_constraints(u_min=ones(2) * -2, u_max=ones(2) * 2)
    xt, ut = trajopt.eval(x_0, max_cvx_iters=20)
    xs, us = true_dyn.simulate(x_0,
                               PiecewiseConstantController(true_dyn, hk, ut),
                               ts)
    true_dyn.plot(xt, ut, ts)
    plt.show()
    xt_gp.max(axis=0)
    xt_gp.min(axis=0)
    #plot trajectories for debugging
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(ts, norm(abs(xt - xs), axis=1), label='True Dynamics')
    ax.plot(ts, norm(abs(xt_gp - xs_gp), axis=1), label='GP Dynamics')
    ax.legend(loc='upper left')
    # plot_2D_q_sys(xt, ax=ax, color='black', box_side=10,
    #               start=x_0[:2], end=x_f[:2])
    # plot_2D_q_sys(xt_gp_im, ax=ax, color='green', box_side=10,
    #               start=x_0[:2], end=x_f[:2])
    # plot_2D_q_sys(xs_gp, ax=ax,  color='red', box_side=10)
    # plot_2D_q_sys(xt_gp, ax=ax, color='blue', box_side=10)
    plt.show()

    # fig = plt.figure()
    # ax = fig.add_subplot()
    # plot_2D_q_sys(xt, ax=ax, color='black', box_side=10, start=(-pi, 0),
    #               end=(0,0))
    # plt.show()
    # true_dyn.plot(xt, us, ts)
    # plt.show()


def test_pend_trajopt():
    m = 1
    l = 0.5
    T = 4
    K = 100
    ts, hk = linspace(0, T, K, retstep=True)
    true_dyn = InvertedPendulum(m, l)
    x_0 = array([-pi, 0.0])

    trajopt = TrajectoryOptimizer(K, hk, true_dyn,
                                  max_delta_u=array([1.2]),
                                  collocation_mode=TrajectoryOptimizer.COLLOCATION_MODE.CTN_ONE_PT,
                                  solver='GUROBI') #GUROBI because OSQP can fail
    trajopt.add_static_quad_cost(Q=array([[1e3, 0], [0, 1e2]]))
    trajopt.add_terminal_cost(array([[1e7, 0], [0, 1e6]]))
    trajopt.add_input_constraints(u_min=ones(1) * -1.2, u_max=ones(1) * 1.2)
    xt, ut = trajopt.eval(x_0, max_cvx_iters=10)
    xs, us  = true_dyn.simulate(x_0,
                                PiecewiseConstantController(true_dyn, hk, ut),ts)
    theta0s = random.uniform(-2*pi, pi/2, (1000,))
    thetadot0s = random.uniform(-8, 8, (1000,))
    u0s = random.uniform(-1.2, 1.2, (1000,1))
    train_x0s = stack([theta0s, thetadot0s], axis=1)
    train_xtp1s = stack([true_dyn.step(x_0=x0, u_0=u0, t_0=0, t_f=hk)
                   for x0, u0 in zip(train_x0s, u0s)], axis=0)
    X = concatenate([train_x0s, u0s], axis=1)
    X = th.from_numpy(X)
    Y =  th.from_numpy(train_xtp1s)
    gp_est = GaussianProcess(X, Y, RBFKernel(2+1))
    gp_est.train_model(n_iters=80)
    gp_dyn = AffineGPSystem(gp_est, n=2, m=1, delta_t=hk)

    gp_trajopt = TrajectoryOptimizer(K, hk, gp_dyn,
                                  max_delta_u=array([1.2]),
                                  collocation_mode=TrajectoryOptimizer.COLLOCATION_MODE.CTN_ONE_PT,
                                  solver='GUROBI')
    gp_trajopt.add_static_quad_cost(Q=array([[1e3, 0], [0, 1e2]]))
    gp_trajopt.add_terminal_cost(array([[1e7, 0], [0, 1e6]]))
    gp_trajopt.add_input_constraints(u_min=ones(1) * -1.2, u_max=ones(1) * 1.2)
    xt_gp, ut_gp = gp_trajopt.eval(x_0, max_cvx_iters=15)
    xs_gp, us_gp = true_dyn.simulate(x_0,
                               PiecewiseConstantController(gp_dyn, hk, ut_gp),
                               ts)
    #TODO: there appear to be spikes in ut_gp - ut. Investigate why.
    #open loop estimation gap is reasonable
    assert norm(xt_gp - xs_gp, axis=1).mean() - norm(xt - xs, axis=1).mean() < 1e-1
    assert (ut_gp - ut).mean() < 1e-1 #controllers should be similar
#plot deviation
    plt.semilogy(ts, norm(xs - xt, axis=1), label="True Model")
    plt.semilogy(ts, norm(xs_gp - xt_gp, axis=1), label="GP Model")
    plt.title('Pendulum System Deviation')
    plt.ylabel('$|| x_t - \hat{x}_t ||_2 $')
    plt.xlabel('$t$')
    plt.legend(loc='upper left')
    plt.show()
    #uncomment to see dynamics paths
    # true_dyn.plot(xs_gp, us_gp, ts)
    # plt.show()

    #uncomment to verify nominal trajectory is reasonable
    # true_dyn.plot(xs, us, ts)
    # plt.show()

    #uncomment to verify dynamics match
    # fig, axs = plt.subplots(1,3, figsize=(18,6))
    # expected, _ = plot_2D_dyn_sys(dyn, axs[0],
    #                               low_x=-pi, high_x=pi, n_sample=100)
    # axs[0].set_title('Expected Phase Plot')
    # actual, _ = plot_2D_dyn_sys(gp_dyn, axs[1],
    #                             low_x=-pi, high_x=pi, n_sample=100)
    # axs[1].set_title('Actual Phase Plot')
    # error = norm(actual - expected,2, axis=2)
    # # assert error.mean() <= 1e-1
    # #uncomment plotting for debugging
    # sns.heatmap(error, ax=axs[2])
    # axs[2].set_title('Error of Phase Plot')
    # plt.show()

def test_4D_lin_trajopt():
    T = 10
    K = 200
    true_A = array([
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [-1, 0, 0, 0],
        [0, 1, 0, 0],
    ])
    true_B = array([
        [0, 0],
        [0, 0],
        [1, 0],
        [0, 1]
    ])
    true_dyn = LinearSystemDynamics(true_A, true_B)
    ts, h_k = linspace(0, T, K, retstep=True)
    x_0 = array([0, 0, .01, 0])
    x_f = array([5, 5, 0, 0])
    trajopt = TrajectoryOptimizer(K, h_k, true_dyn,
                                  TrajectoryOptimizer.COLLOCATION_MODE.CTN_ONE_PT)
    trajopt.add_static_quad_cost()
    trajopt.add_terminal_cost(Qf=eye(4) * 1e5, xf=x_f)
    trajopt.add_input_constraints(ones(2) * -9, ones(2) * .95)
    [xt_mpc, ut] = trajopt.eval(x_0, max_cvx_iters=1)
    xs, us = true_dyn.simulate(
        x_0,
        PiecewiseConstantController(true_dyn, trajopt.h_k, ut),
        ts)
    train_x0s = random.uniform(-7.5, 7.5, (1000, 4))
    train_u0s = random.uniform(-9, 1, (1000, 2))
    X = concatenate([train_x0s, train_u0s], axis=1)
    Y = stack([true_dyn.step(x, u, t_0=0, t_f=h_k)
               for x,u in zip(train_x0s, train_u0s)], axis=0)
    X = th.from_numpy(X)
    Y = th.from_numpy(Y)

    gp_est = GaussianProcess(X, Y, RBFKernel(4+2))
    gp_est.train_model(n_iters=300)
    gp_system = AffineGPSystem(gp_est, 4, 2, h_k)
    gp_trajopt = GPTrajectoryOptimizer(K, h_k, gp_system)
    gp_trajopt.add_static_quad_cost()
    gp_trajopt.add_terminal_cost(Qf=eye(4) * 1e5, xf=x_f)
    gp_trajopt.add_input_constraints(ones(2) * -9, ones(2) * 1)
    [xt_gp, ut_gp] = gp_trajopt.eval(x_0, max_cvx_iters=1)
    xs_gp, us_gp = true_dyn.simulate(x_0,
                                     PiecewiseConstantController(true_dyn, h_k, ut_gp),
                                     ts)
    xt_gp_im, ut_gp_im = gp_system.simulate(x_0,
                                           PiecewiseConstantController(
                                               gp_system, h_k, ut_gp),
                                           ts)

    assert norm((xs_gp - xs),axis=1).mean() < 1e-1
    assert norm(ut_gp - ut, axis=1).mean() < 1e-1
    #plot deviation
    # plt.semilogy(ts, norm(xs - xt_mpc, axis=1), label="True Model")
    # plt.semilogy(ts, norm(xs_gp - xt_gp, axis=1), label="GP Model")
    # plt.title('Linear System Deviation')
    # plt.ylabel('$|| x_t - \hat{x}_t ||_2 $')
    # plt.xlabel('$t$')
    # plt.legend(loc='upper left')
    # plt.show()
    #plot trajectories for debugging
    # fig = plt.figure()
    # ax = fig.add_subplot()
    # plot_2D_q_sys(xs, ax=ax, color='black', box_side=10)
    # # plot_2D_q_sys(xt_gp_im, ax=ax, color='red', box_side=100)
    # plot_2D_q_sys(xs_gp, ax=ax,  color='red', box_side=10)
    # plot_2D_q_sys(xt_gp, ax=ax, color='blue', box_side=10)
    # plt.show()

def test_2D_lin_trajopt():
    K = 200
    T = 10
    max_abs_delta_u = 10
    ts, h_k = linspace(0, T, K, retstep=True)
    true_A = array([[0, 1],
                    [1, 0]])
    true_B = array([[0],
                    [1]])

    x_0 = array([0, -0.0005])
    x_f = array([10, 0])

    true_dyn = LinearSystemDynamics(true_A, true_B)


    trajopt = TrajectoryOptimizer(K, h_k, true_dyn,
                                  TrajectoryOptimizer.COLLOCATION_MODE.CTN_ONE_PT)
    trajopt.add_static_quad_cost()
    trajopt.add_terminal_cost(Qf=eye(2) * 1e5, xf=x_f)
    trajopt.add_input_constraints(ones(1) * -max_abs_delta_u, ones(1) *  max_abs_delta_u)
    [xt_mpc, ut] = trajopt.eval(x_0, max_cvx_iters=1)
    xs, us = true_dyn.simulate(x_0,
                            PiecewiseConstantController(true_dyn,
                                                        trajopt.h_k, ut),
                            ts)

    # X0s, V0s, U0s = meshgrid(
    #     linspace(-20, 20, 10),
    #     linspace(-10, 10, 10),
    #     linspace(-max_abs_delta_u, max_abs_delta_u, 10))
    # train_x0s = stack([X0s.ravel(), V0s.ravel()], axis=1)
    # train_u0s = U0s.ravel()[:,newaxis]
    train_x0s = (random.rand(1000, 2) - 0.5) * 20
    train_u0s = (random.rand(1000, 1) - 0.5) * 2 * max_abs_delta_u

    X = concatenate([train_x0s, train_u0s], axis=1)
    Y = stack([true_dyn.step(x, u, t_0=0, t_f=h_k)
               for x,u in zip(train_x0s, train_u0s)], axis=0)
    X = th.from_numpy(X)
    Y = th.from_numpy(Y)
    gp_est = GaussianProcess(X, Y, RBFKernel(2+1))
    gp_est.train_model(n_iters=300)
    gp_system = AffineGPSystem(gp_est, 2, 1, h_k)
    gp_trajopt = GPTrajectoryOptimizer(K, h_k, gp_system)
    gp_trajopt.add_static_quad_cost()
    gp_trajopt.add_terminal_cost(Qf=eye(2) * 1e5, xf=x_f)
    gp_trajopt.add_input_constraints(ones(1) * -max_abs_delta_u,
                                  ones(1) * max_abs_delta_u)
    [xt_gp, ut_gp] = gp_trajopt.eval(x_0, max_cvx_iters=1)
    xs_gp, us_gp = true_dyn.simulate(x_0,
                                     PiecewiseConstantController(true_dyn, h_k, ut_gp),
                                     ts)
    #gp trajectory prediction should be close to true trajectory
    assert norm(xs_gp - xt_gp, axis=1).mean() < 1e-1
    #gp commands should be close to commands from true system
    assert abs(ut_gp - ut).max() < 1e-1

    #plot deviation
    # plt.semilogy(ts, norm(xs - xt_mpc, axis=1), label="True Model")
    # plt.semilogy(ts, norm(xs_gp - xt_gp, axis=1), label="GP Model")
    # plt.title('Linear System Deviation')
    # plt.ylabel('$|| x_t - \hat{x}_t ||_2 $')
    # plt.xlabel('$t$')
    # plt.legend(loc='upper left')
    # plt.show()
    low_x = -20
    high_x = 20
    low_y = -10
    high_y = 10
    #debugging plots for GP phase approximation
    # fig, axs = plt.subplots(1,3)
    # expected, _ = plot_2D_dyn_sys(true_dyn, axs[0], n_sample=100,
    #                               low_x=low_x, high_x=high_x,
    #                               low_y=low_y, high_y=high_y)
    # axs[0].set_title('Expected Phase Plot')
    # actual, _ = plot_2D_dyn_sys(gp_system, axs[1], n_sample=100,
    #                             low_x=low_x, high_x=high_x,
    #                             low_y=low_y, high_y=high_y)
    # axs[1].set_title('Actual Phase Plot')
    # error = norm(actual - expected,2, axis=2)
    # sns.heatmap(error, ax=axs[2])
    # axs[2].set_title('Error of Phase Plot')
    # plt.show()
    # debugging plots for trajectory comparisons
    # fig, axs = plt.subplots(2, 2)
    # axs[0,0].plot(xs[:, 0], xs[:, 1], color='black', label="Simulated Path")
    # axs[0,0].plot(xt_mpc[:,0], xt_mpc[:,1], color='red', label="MPC Path")
    # axs[0,0].legend(loc='upper left')
    # axs[0,0].set_xlim(left=low_x, right=high_x)
    # axs[0,0].set_ylim(bottom=low_y, top=high_y)
    # axs[0,0].set_xlabel("Position")
    # axs[0,0].set_ylabel("Velocity")
    # start_circ = plt.Circle(x_f, 0.3, color='g')
    # end_circ = plt.Circle(x_0, 0.3, color='b')
    # axs[0,0].add_patch(start_circ)
    # axs[0,0].add_patch(end_circ)
    # axs[0,0].set_title("State Space")
    # axs[0,1].plot(ts[:-1], us)
    # axs[0,1].set_title("Action Space")
    # axs[0,1].set_xlabel("Time")
    # axs[0,1].set_ylabel("u")
    #
    # axs[1,0].plot(xs_gp[:, 0], xs_gp[:, 1], color='black', label="Simulated Path")
    # axs[1,0].plot(xt_gp[:, 0], xt_gp[:,1], color='red', label="MPC Path")
    # axs[1,0].legend(loc='upper left')
    # axs[1,0].set_xlim(left=low_x, right=high_x)
    # axs[1,0].set_ylim(bottom=low_y, top=high_y)
    # axs[1,0].set_xlabel("Position")
    # axs[1,0].set_ylabel("Velocity")
    # start_circ = plt.Circle(x_f, 0.3, color='g')
    # end_circ = plt.Circle(x_0, 0.3, color='b')
    # axs[1,0].add_patch(start_circ)
    # axs[1,0].add_patch(end_circ)
    # axs[1,0].set_title("State Space")
    # axs[1,1].plot(ts[:-1], us_gp)
    # axs[1,1].set_title("Action Space")
    # axs[1,1].set_xlabel("Time")
    # axs[1,1].set_ylabel("u")
    # plt.show()