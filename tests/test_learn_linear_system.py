from numpy import concatenate, eye, linspace, pi, ones, random, \
    sin, cos, newaxis, stack, meshgrid
from numpy.core._multiarray_umath import array, zeros
from scipy.linalg import expm, logm
from numpy.linalg import norm, pinv
from numpy.testing import assert_allclose
from core.learning import GaussianProcess, RBFKernel, PeriodicKernel, \
    AdditiveKernel, ScaledGaussianProcess, GPScaler, MultiplicativeKernel
from core.systems import LinearSystemDynamics, InvertedPendulum, AffineGPSystem
from core.controllers import ConstantController, PiecewiseConstantController
from core.trajopt import TrajectoryOptimizer
from torch.autograd.functional import jacobian
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns

import gpytorch as gp
import torch as th

th.set_default_dtype(th.float64)
random.seed(0)
th.manual_seed(0)

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

linear_dyn = LinearSystemDynamics(true_A, true_B)

T = 10
K = 50


def plot_2D_q_sys(xs, ax=None, color='black', box_side=60):
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
                 length_includes_head=True,
                 color=color)
    ax.set_xlim(-box_side, box_side)
    ax.set_ylim(-box_side, box_side)
    ax.add_artist(end_circ)
    ax.add_artist(start_circ)
    return ax


def generate_optimal_trajectory(dynamics):
    ts, h_k = linspace(0, T, K, retstep=True)
    x_0 = array([0, 0, 10, 0])
    x_f = array([50, 50, 0, 0])
    trajopt = TrajectoryOptimizer(K, h_k, dynamics,
                                  TrajectoryOptimizer.COLLOCATION_MODE.CTN_ONE_PT)
    trajopt.add_static_quad_cost()
    # trajopt.add_static_quad_cost(Q=eye(4) * 1e3, offset=x_f)
    # trajopt.add_static_quad_cost(Q=eye(4) * 1e4, offset=x_f)
    trajopt.add_terminal_cost(Q_f=eye(4) * 1e5, offset=x_f)
    trajopt.add_input_constraints(ones(2) * -50, ones(2) * 1)
    # can't do hard terminal constraint with very infeasible models
    [xt_mpc, ut] = trajopt.eval(x_0, max_cvx_iters=1)

    xut = linear_dyn.simulate(
        x_0,
        PiecewiseConstantController(dynamics, trajopt.h_k, ut),
        ts)
    fig, axs = plt.subplots(1,2)
    plot_2D_q_sys(xut[0], box_side=100, ax=axs[0])
    plot_2D_q_sys(xt_mpc, color='red', box_side=100, ax= axs[1])
    plt.show()
    return xut + (h_k,)


def plot_2D_dyn_sys(dyn,  ax=None,
                    low_x=-5, high_x=5,
                    low_y=-5, high_y=5,
                    n_sample=100):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot()
    X, Y = meshgrid(linspace(low_x, high_x, n_sample),
                    linspace(low_y, high_y, n_sample))
    arrows = array([dyn.eval_dot(stack(xy), zeros((1,)), 0) for xy in
                    zip(X.ravel(), Y.ravel())])
    arrows = arrows.reshape(X.shape + (2,))
    ax.streamplot(X, Y, arrows[:, :, 0], arrows[:, :, 1])
    return arrows, ax


def  XY_from_traj(xt, ut):
    xt = th.from_numpy(xt)
    ut = th.from_numpy(ut)
    X = th.cat([xt[:-1], ut], dim=1).clone()
    Y = xt[1:].contiguous()
    return X, Y


def test_rl_linear_system():
    n = 4
    m = 2
    xut_opt = generate_optimal_trajectory(linear_dyn)
    A_init = true_A + random.rand(4, 4) * 0.05
    B_init = true_B + random.rand(4, 2) * 0.05

    max_iter = 5
    A = A_init
    B = B_init
    Aerrs = list()
    Berrs = list()
    path_errors = list()
    xt, ut, hk = generate_optimal_trajectory(LinearSystemDynamics(A, B))
    gp_est = None
    for i in range(max_iter):
        X, Y = XY_from_traj(xt, ut)
        if gp_est is None:
            gp_est = GaussianProcess(X, Y, RBFKernel(n+m))
            gp_est.train_model(100)
        else:
            gp_est.add_samples(X, Y)
            gp_est.train_model(10)

        xt, ut, hk = generate_optimal_trajectory(
            AffineGPSystem(gp_est, n=n, m=m, delta_t=hk))
        # mu_prime, cov_prime = gp_est.ddx(X)



def test_gp_lin_dyn_trajopt():
    dyn = LinearSystemDynamics(true_A, true_B)
    n_samples = 100
    train_x0s = (random.rand(n_samples, 4) - 0.5) * 80
    train_u0s = (random.rand(n_samples, 2) - 0.5) * 50
    dt = 0.01
    train_xtp1s = [dyn.step(x_0=x0, u_0=u0, t_0=0, t_f=dt) for x0, u0 in zip(train_x0s, train_u0s)]
    train_xtp1s = stack(train_xtp1s, axis=0)
    train_X = concatenate((train_x0s, train_u0s), axis=1)
    train_X = th.from_numpy(train_X)
    train_Y =  th.from_numpy(train_xtp1s)

    x_scaler = GPScaler(xmins=th.tensor([-40.] * 4 + [-25] * 2),
                        xmaxs=th.tensor([40.] * 4 + [25]* 2))
    y_scaler = GPScaler(xmins=th.tensor([-40.] * 4),
                        xmaxs=th.tensor([40.] * 4))
    gp_est = ScaledGaussianProcess(train_X, train_Y, RBFKernel(4+2),
                                   x_scaler, y_scaler)
    # gp_est = GaussianProcess(train_X, train_Y, RBFKernel(4+2))
    gp_est.train_model(n_iters=160, lr=0.1)
    gp_dyn = AffineGPSystem(gp_est, n=4, m=2, delta_t=dt)
    xt_true, ut_true, hk_true = generate_optimal_trajectory(dyn)
    xt_gp, ut_gp, hk_gp = generate_optimal_trajectory(gp_dyn)
    error = xt_true - xt_gp
    assert norm(error, axis=-1).mean() < 1e-1

def test_pendulum_periodic_kernel_phase():
    m = 1
    l = 0.5

    dyn = InvertedPendulum(m, l)
    random.randn()
    train_x0s = (random.rand(100, 2) - 0.5) * array([2*pi, 10])
    u_0 = array([0.0])
    dt = 0.01
    train_xtp1s = [dyn.step(x_0=x0, u_0=u_0, t_0=0, t_f=dt) for x0 in train_x0s]
    train_xtp1s = stack(train_xtp1s, axis=0)
    train_X = concatenate((train_x0s, zeros((train_x0s.shape[0], 1))), axis=1)
    train_X = th.from_numpy(train_X)
    train_Y =  th.from_numpy(train_xtp1s)
    train_Y = train_Y - train_X[:,:2]
    x_scaler = GPScaler(th.tensor([-pi, -5, -1]), th.tensor([pi, 5, 1]))
    y_scaler = GPScaler(th.tensor([-pi, -5]), th.tensor([pi, 5]))
    from numpy import sqrt
    # kernel = MultiplicativeKernel(
    #     kernels=[PeriodicKernel(p_prior=2.,
    #
    #                             learn_period=False),
    #              RBFKernel(1, ard_num_dims=True)],
    #     active_dims=[[0], [1]]
    # )
    # X, Y = meshgrid(linspace(-4, 4, 1000),
    #                 linspace(-4, 4, 1000))
    # errors = array([
    #     kernel(th.from_numpy(stack(xy))[None,:], th.zeros(1,2)).detach().numpy()
    #     for
    #     xy in
    #                 zip(X.ravel(), Y.ravel())])
    # errors = errors.reshape(X.shape)
    # sns.heatmap(errors)
    # plt.show()
    kernel = MultiplicativeKernel(
        kernels=[PeriodicKernel(p_prior=2. * pi,
                                learn_period=False),
                 # RBFKernel(1),
                 # PeriodicKernel(p_prior=2., learn_period=True),
                 RBFKernel(2)
                 ],
        active_dims=[[0],
                     # [0],
                     [1,2],
                     # [1]
                     ]
    )
    gp_est = GaussianProcess(train_X, train_Y, kernel)
    gp_est.train_model(n_iters=55)
    # gp_est.train_model(n_iters=310, lr=0.5)


    # gp_est.train_model(n_iters=500, lr=0.1)
    gp_dyn = AffineGPSystem(gp_est, n=2, m=1, delta_t=dt, force_delta_mode=True)
    # plt.scatter(train_x0s[:,0], train_x0s[:,1], color='blue', marker='o')
    # plt.scatter(train_xtp1s[:, 0], train_xtp1s[:, 1], color='red', marker='o')
    # plt.title('Data Set')
    # plt.show()
    fig, axs = plt.subplots(1,3, figsize=(9,3), dpi=400)
    expected, _ = plot_2D_dyn_sys(dyn, axs[0],
                                  # low_x=-pi, high_x=pi,
                                  low_x=-4*pi, high_x=4*pi,
                                  n_sample=10)
    axs[0].set_title('Expected Phase Plot')
    actual, _ = plot_2D_dyn_sys(gp_dyn, axs[1],
                                # low_x=-pi, high_x=pi,
                                low_x=-4*pi, high_x=4*pi,
                                n_sample=10)
    axs[1].set_title('Actual Phase Plot')
    error = norm(actual - expected,2, axis=2)
    assert error.mean() <= 1e-3
    #uncomment plotting for debugging
    # sns.heatmap(error, ax=axs[2])
    # axs[2].set_title('Error of Phase Plot')
    # plt.show()

def test_pendulum_learned_phase():
    m = 1
    l = 0.5

    dyn = InvertedPendulum(m, l)
    random.randn()
    train_x0s = (random.rand(100, 2) - 0.5) * array([2*pi, 10])
    u_0 = array([0.0])
    dt = 0.01
    train_xtp1s = [dyn.step(x_0=x0, u_0=u_0, t_0=0, t_f=dt) for x0 in train_x0s]
    train_xtp1s = stack(train_xtp1s, axis=0)
    train_X = concatenate((train_x0s, zeros((train_x0s.shape[0], 1))), axis=1)
    train_X = th.from_numpy(train_X)
    train_Y =  th.from_numpy(train_xtp1s)

    gp_est = GaussianProcess(train_X, train_Y, RBFKernel(2+1, ard_num_dims=True))
    gp_est.train_model(n_iters=45)
    gp_dyn = AffineGPSystem(gp_est, n=2, m=1, delta_t=dt)
    # plt.scatter(train_x0s[:,0], train_x0s[:,1], color='blue', marker='o')
    # plt.scatter(train_xtp1s[:, 0], train_xtp1s[:, 1], color='red', marker='o')
    # plt.title('Data Set')
    # plt.show()
    fig, axs = plt.subplots(1,3, figsize=(9,3), dpi=200)
    expected, _ = plot_2D_dyn_sys(dyn, axs[0],
                                  low_x=-pi, high_x=pi, n_sample=100)
    axs[0].set_title('Expected Phase Plot')
    actual, _ = plot_2D_dyn_sys(gp_dyn, axs[1],
                                low_x=-pi, high_x=pi, n_sample=100)
    axs[1].set_title('Actual Phase Plot')
    error = norm(actual - expected,2, axis=2)
    assert error.mean() <= 1e-1
    #uncomment plotting for debugging
    # sns.heatmap(error, ax=axs[2])
    # axs[2].set_title('Error of Phase Plot')
    # plt.show()

def test_pendulum_learned_phase_delta():
    m = 1
    l = 0.5

    dyn = InvertedPendulum(m, l)
    random.randn()
    train_x0s = (random.rand(100, 2) - 0.5) * array([2*pi, 10])
    u_0 = array([0.0])
    dt = 0.01
    train_xtp1s = [dyn.step(x_0=x0, u_0=u_0, t_0=0, t_f=dt) for x0 in train_x0s]
    train_xtp1s = stack(train_xtp1s, axis=0)
    train_X = concatenate((train_x0s, zeros((train_x0s.shape[0], 1))), axis=1)
    train_X = th.from_numpy(train_X)
    train_Y =  th.from_numpy(train_xtp1s)
    train_Y = train_Y - train_X[:,:2]

    gp_est = GaussianProcess(train_X, train_Y,
                             # MultiplicativeKernel(
                             #     kernels=[RBFKernel(2, ard_num_dims=True),
                             #              PeriodicKernel(2. * pi,
                             #                             learn_period=False)],
                             #     active_dims=[[1,2], [0]])
                             RBFKernel(2+1, ard_num_dims=True)
    )
    gp_est.train_model(n_iters=55)
    gp_dyn = AffineGPSystem(gp_est, n=2, m=1, delta_t=dt, force_delta_mode=True)
    # plt.scatter(train_x0s[:,0], train_x0s[:,1], color='blue', marker='o')
    # plt.scatter(train_xtp1s[:, 0], train_xtp1s[:, 1], color='red', marker='o')
    # plt.title('Data Set')
    # plt.show()
    fig, axs = plt.subplots(1,3, figsize=(9,3), dpi=200)
    expected, _ = plot_2D_dyn_sys(dyn, axs[0],
                                  low_x=-pi, high_x=pi, n_sample=10)
    axs[0].set_title('Expected Phase Plot')
    actual, _ = plot_2D_dyn_sys(gp_dyn, axs[1],
                                low_x=-pi, high_x=pi, n_sample=10)
    axs[1].set_title('Actual Phase Plot')
    error = norm(actual - expected,2, axis=2)
    assert error.mean() <= 1e-3
    #uncomment plotting for debugging
    # sns.heatmap(error, ax=axs[2],
    #             # norm=LogNorm(error.min(), error.max())
    #             )
    # axs[2].set_title('Error of Phase Plot')
    # plt.show()

def test_pendulum_learned_phase_delta_kin_approx():
    m = 1
    l = 0.5

    dyn = InvertedPendulum(m, l)
    random.randn()
    train_x0s = (random.rand(100, 2) - 0.5) * array([2*pi, 10])
    u_0 = array([0.0])
    dt = 0.01
    train_xtp1s = [dyn.step(x_0=x0, u_0=u_0, t_0=0, t_f=dt) for x0 in train_x0s]
    train_xtp1s = stack(train_xtp1s, axis=0)
    train_X = concatenate((train_x0s, zeros((train_x0s.shape[0], 1))), axis=1)
    train_X = th.from_numpy(train_X)
    train_Y =  th.from_numpy(train_xtp1s)
    train_Y = train_Y - train_X[:,:2]

    gp_est = GaussianProcess(train_X, train_Y[:,1:],
                             # MultiplicativeKernel(
                             #     kernels=[RBFKernel(2, ard_num_dims=True),
                             #              PeriodicKernel(2. * pi,
                             #                             learn_period=False)],
                             #     active_dims=[[1,2], [0]])
                             RBFKernel(2+1, ard_num_dims=True)
    )
    gp_est.train_model(n_iters=55)
    gp_dyn = AffineGPSystem(gp_est, n=2, m=1, delta_t=dt,
                            force_delta_mode=True,
                            ddim_to_dim={1: 0},
                            ddim_to_gp_idx={1: 0})
    # plt.scatter(train_x0s[:,0], train_x0s[:,1], color='blue', marker='o')
    # plt.scatter(train_xtp1s[:, 0], train_xtp1s[:, 1], color='red', marker='o')
    # plt.title('Data Set')
    # plt.show()

    fig, axs = plt.subplots(1,3, figsize=(9,3), dpi=200)
    expected, _ = plot_2D_dyn_sys(dyn, axs[0],
                                  low_x=-pi, high_x=pi, n_sample=10)
    axs[0].set_title('Expected Phase Plot')
    actual, _ = plot_2D_dyn_sys(gp_dyn, axs[1],
                                low_x=-pi, high_x=pi, n_sample=10)
    axs[1].set_title('Actual Phase Plot')
    error = norm(actual - expected,2, axis=2)
    assert error.mean() <= 1e-1
    #uncomment plotting for debugging
    sns.heatmap(error, ax=axs[2])
    axs[2].set_title('Error of Phase Plot')
    plt.show()

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
    gp_est.train_model(n_iters=40)
    gp_dyn = AffineGPSystem(gp_est, n=2, m=1, delta_t=dt)

    # plt.scatter(train_x0s[:,0], train_x0s[:,1], color='blue', marker='o')
    # plt.scatter(train_xtp1s[:, 0], train_xtp1s[:, 1], color='red', marker='o')
    # plt.title('Data Set')
    # plt.show()
    fig, axs = plt.subplots(1,3, figsize=(9,3), dpi=200)
    expected, _ = plot_2D_dyn_sys(dyn, axs[0], n_sample=10)
    axs[0].set_title('Expected Phase Plot')
    actual, _ = plot_2D_dyn_sys(gp_dyn, axs[1], n_sample=10)
    axs[1].set_title('Actual Phase Plot')
    error = norm(actual - expected,2, axis=2)
    assert error.mean() <= 1e-1
    #uncomment plotting for debugging
    # sns.heatmap(error, ax=axs[2])
    # axs[2].set_title('Error of Phase Plot')
    # plt.show()

def test_2D_lin_point_jacobian():
    true_A = array([
        [0, -3],
        [1, 0]
    ])
    true_B = array([
        [0], [1]])

    dyn = LinearSystemDynamics(true_A, true_B)

    ts, hk = linspace(0, 10, 200, retstep=True)
    xs_train, us_train = dyn.simulate(array([0, 1]),
                                      ConstantController(dyn, zeros((1,))),
                                      ts=ts)
    xs_test, us_test = dyn.simulate(array([0.01, 0.99]),
                                    # array([cos(0.1), sin(0.1)]),
                                    ConstantController(dyn, zeros((1,))),
                                    ts=ts)

    train_x = th.from_numpy(concatenate([xs_train[:-1, :], us_train], axis=1))
    train_y = th.from_numpy(xs_train[1:, :])

    test_x = th.from_numpy(concatenate([xs_test[:-1, :], us_test], axis=1))
    test_y = th.from_numpy(xs_test[1:, :])

    gpdyn = GaussianProcess(train_x, train_y, RBFKernel(2+1))
    gpdyn.train_model(n_iters=550, lr=0.03)
    mu, cov = gpdyn(test_x)
    mu_prime, cov_prime = gpdyn.ddx(test_x)
    mu = mu.detach().numpy()
    cov = cov.detach().numpy()
    mu_prime = mu_prime.detach().numpy()
    cov_prime = cov_prime.detach().numpy()
    expAts = mu_prime[:, :, :2]

    As = stack([logm(expAt).real * (1/hk) for expAt in expAts], axis=0)
    #checks if pointise jacobians are accurate
    assert_allclose(As, true_A[newaxis].repeat(As.shape[0], axis=0), atol=1e-5)