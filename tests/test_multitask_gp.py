from numpy import pi

from core.learning.gaussian_process import RBFKernel, GaussianProcess, \
    GPScaler, ScaledGaussianProcess, PeriodicKernel, AdditiveKernel, \
    AffineDotProductKernel, MultiplicativeKernel, save_gp, load_gp
import matplotlib.pyplot as plt
import gpytorch as gpt
import torch as th
import pathlib
from os import makedirs

th.manual_seed(0)

def test_1dx1dgp():
    sigma = 0.2
    train_x = th.rand((20,1))*2*pi
    train_y = th.sin(train_x) + sigma * (th.rand_like(train_x) - 0.5)

    kernel = RBFKernel(1)
    gp = GaussianProcess(train_x, train_y, kernel)
    gp.train_model()

    test_x = th.linspace(0, 2 * pi, 1000).unsqueeze(1)
    test_y = th.sin(test_x)

    y_hat, cov = gp(test_x)

    out_dist = gpt.distributions.MultivariateNormal(y_hat.squeeze(), cov.to_dense().squeeze())
    lower, upper = out_dist.confidence_region()
    assert (y_hat - test_y).abs().mean().item() < 1e-1
    assert (test_y.squeeze() > lower).all()
    assert (upper > test_y.squeeze()).all()
    #uncomment plotting for debugging
    # plt.plot(train_x.squeeze().detach().numpy(),
    #          train_y.detach().numpy(), 'o',
    #         color='black',
    #         markersize=5,
    #         fillstyle="none")
    # plt.plot(test_x.detach().squeeze().numpy(),
    #          test_y.detach().numpy(),
    #          label='True Y')
    # plt.fill_between(test_x.detach().squeeze().numpy(),
    #                 lower.detach().numpy(),
    #                 upper.detach().numpy(), alpha=0.5)
    # plt.plot(test_x.detach().squeeze().numpy(),
    #          y_hat.detach().numpy(), label='Estimate')
    # plt.legend(loc='upper right')
    # plt.show()


def test_2dx1dgp():
    sigma = 0.2
    train_X = th.rand((650,2))*2*pi
    train_Y = th.sin(train_X[:,0])*th.cos(train_X[:,1]) \
              + sigma * (th.rand_like(train_X[:,0]) - 0.5)
    train_Y = train_Y.unsqueeze(1)
    kernel = RBFKernel(2)
    gp = GaussianProcess(train_X, train_Y, kernel)
    gp.train_model()
    dim_sample = th.linspace(0,2*pi, 100)
    X, Y = th.meshgrid(dim_sample, dim_sample)
    x_vec, y_vec = th.flatten(X), th.flatten(Y)
    test_x = th.stack((x_vec, y_vec), dim=1)
    y_hat, cov = gp(test_x)
    out_dist = gpt.distributions.MultivariateNormal(y_hat.squeeze(), cov.to_dense().squeeze())
    lower, upper = out_dist.confidence_region()
    test_y =  (th.sin(test_x[:,0])*th.cos(test_x[:,1])).unsqueeze(1)

    assert (y_hat - test_y).abs().max().item() < sigma
    assert (test_y.squeeze() > lower).all()
    assert (upper > test_y.squeeze()).all()
    #uncomment plotting for debugging
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(X.detach().numpy(), Y.detach().numpy(),
    #                 (y_hat - test_y).abs().view_as(X).detach().numpy())
    # # ax.plot_surface(X.detach().numpy(), Y.detach().numpy(), y_hat.view_as(X).detach().numpy())
    # plt.show()

def test_gp_save_load():
    sigma = 0.2
    train_X = th.rand((650,2))*2*pi
    train_Y = th.sin(train_X[:,0])*th.cos(train_X[:,1]) \
              + sigma * (th.rand_like(train_X[:,0]) - 0.5)
    train_Y = train_Y.unsqueeze(1)
    kernel = RBFKernel(2)
    gp_original = GaussianProcess(train_X, train_Y, kernel)
    gp_original.train_model()

    root_dir = pathlib.Path(__file__).parent.absolute()
    save_dir = root_dir / 'data' / 'test_load_save_gp'
    makedirs(save_dir, exist_ok=True)

    save_file = save_dir / 'test_save.th'
    save_gp(gp_original, save_file)
    gp_loaded = load_gp(save_file, RBFKernel(2))

    dim_sample = th.linspace(0,2*pi, 100)
    X, Y = th.meshgrid(dim_sample, dim_sample)
    x_vec, y_vec = th.flatten(X), th.flatten(Y)
    test_x = th.stack((x_vec, y_vec), dim=1)
    y_hat, cov = gp_loaded(test_x)
    out_dist = gpt.distributions.MultivariateNormal(y_hat.squeeze(), cov.to_dense().squeeze())
    lower, upper = out_dist.confidence_region()
    test_y =  (th.sin(test_x[:,0])*th.cos(test_x[:,1])).unsqueeze(1)

    assert (y_hat - test_y).abs().max().item() < sigma
    assert (test_y.squeeze() > lower).all()
    assert (upper > test_y.squeeze()).all()

def test_2d2dgp_multiplicative_periodic():
    sigma = 0.5
    train_X = th.stack([th.distributions.Uniform(0, 2 * pi * 10).sample((250,)),
                        th.distributions.Uniform(-10 * pi * 100, 10 * pi * 100).sample(
                            (250,))], dim=1)

    def genYYprime(X):
        Y = th.stack([
            10 * th.sin(X[:, 0]/10),
            10 * th.cos(X[:, 0]/10) * X[:, 1]], dim=1)
        Y_prime = th.stack([
            th.stack([th.cos(X[:, 0] / 10), th.zeros((X.shape[0], ))], dim=1),
            th.stack([(- th.sin(X[:, 0] / 10) * X[:,1]),
                      10 * th.cos(X[:, 0] / 10)], dim=1)], dim=1)
        return Y, Y_prime

    train_Y, train_Y_prime = genYYprime(train_X)
    train_Y += th.distributions.Normal(0.0, sigma).sample(train_X.shape)

    # kernel = RBFKernel(2, ard_num_dims=True)
    kernel = MultiplicativeKernel(
                kernels=[PeriodicKernel(2), RBFKernel(1)],
                active_dims=[[0], [1]])
    x_scaler = GPScaler(xmins=th.tensor([0, -10 * pi * 100]),
                        xmaxs=th.tensor([2 * pi * 10, 10 * pi * 100]),
                        wraps=th.tensor([True, False]))
    y_scaler = GPScaler(xmins=th.tensor([-10, -100 * pi * 100]),
                        xmaxs=th.tensor([10, 100 * pi * 100]))
    gp = ScaledGaussianProcess(train_X, train_Y, kernel,
                               x_scaler=x_scaler, y_scaler=y_scaler)
    gp.train_model()
    x_vec = th.linspace(-10 * pi * 10, 10 * pi * 10, 100)
    y_vec = th.linspace(-10 * pi * 100, 10 * pi * 100, 100)
    test_x = th.stack((x_vec, y_vec), dim=1)
    test_y , test_y_prime = genYYprime(test_x)
    y_hat, cov = gp(test_x)
    y_hat_prime, cov_prime = gp.ddx(test_x)
    out_dist_1 = gpt.distributions.MultivariateNormal(y_hat.squeeze()[:, 0],
                                                      cov[:, :, 0, 0])
    out_dist_2 = gpt.distributions.MultivariateNormal(y_hat.squeeze()[:, 1],
                                                      cov[:, :, 1, 1])
    lower1, upper1 = out_dist_1.confidence_region()
    lower2, upper2 = out_dist_2.confidence_region()
    lower1_p, upper1_p = gpt.distributions.MultivariateNormal(
        y_hat_prime[:, 0, 0], cov_prime[:, :, 0, 0, 0, 0]).confidence_region()
    lower2_p, upper2_p = gpt.distributions.MultivariateNormal(
        y_hat_prime[:, 1, 1], cov_prime[:, :, 1, 1, 1, 1]).confidence_region()

    # these dont' guarantee true value is in confidence region but its good
    # enough
    # GP tests
    assert (test_y[:, 0] > lower1).all()
    assert (test_y[:, 1] > lower2).all()
    assert (test_y[:, 0] < upper1).all()
    assert (test_y[:, 1] < upper2).all()
    #big error because the problem is scaled to be very large
    assert (y_hat[5:-5,:] - test_y[5:-5,:]).abs().mean() < 25
    # # derivative of GP tests
    assert abs(y_hat_prime[:, 0, 0] - test_y_prime[:, 0, 0]).mean() < 1
    assert abs(y_hat_prime[:, 1, 1] - test_y_prime[:, 1, 1]).mean() < 1
    assert abs(y_hat_prime[:, 1, 0] - test_y_prime[:, 1, 0]).mean() < 10
    assert y_hat_prime[:, 0,  1].abs().mean() < 1  # notice the loss of a significant figure

    assert (y_hat_prime[:, 0, 0] < upper1_p).all()
    assert (y_hat_prime[:, 0, 0] > lower1_p).all()
    assert (y_hat_prime[:, 1, 1] < upper2_p).all()
    assert (y_hat_prime[:, 1, 1] > lower2_p).all()

    # uncomment plots for debugging
    # f, axs = plt.subplots(2,2, dpi=200)
    # axs[0][0].plot(test_x[:,0].detach().squeeze().numpy(),
    #             y_hat[:,0].detach().squeeze().numpy(), label='Estimate')
    # axs[0][0].plot(test_x[:,0].detach().squeeze().numpy(),
    #             test_y[:,0].detach().squeeze().numpy(), label='True Y')
    # axs[0][0].legend(loc='upper right')
    # axs[0][0].plot(train_X[:,0].squeeze().detach().numpy(),
    #             train_Y[:,0].detach().numpy(), 'o',
    #         color='black',
    #         markersize=5,
    #         fillstyle="none")
    # axs[0][0].fill_between(test_x[:,0].detach().squeeze().numpy(),
    #                     lower1.detach().numpy(),
    #                     upper1.detach().numpy(), alpha=0.5)
    # axs[0][0].title.set_text('$10 \\sin(\\frac{x_1}{10})$')
    # axs[0][1].plot(test_x[:,1].detach().squeeze().numpy(),
    #             y_hat[:,1].detach().squeeze().numpy(), label='Estimate')
    # axs[0][1].plot(test_x[:, 1].detach().squeeze().numpy(),
    #             test_y[:, 1].detach().squeeze().numpy(), label='True Y')
    # axs[0][1].legend(loc='upper right')
    # axs[0][1].fill_between(test_x[:, 1].detach().squeeze().numpy(),
    #                     lower2.detach().numpy(),
    #                     upper2.detach().numpy(), alpha=0.5)
    # axs[0][1].title.set_text('$10 \\cos(\\frac{x_1}{10}) x_2$')
    # axs[1][0].plot(test_x[:, 0].detach().squeeze().numpy(),
    #             y_hat_prime[:, 0,0].detach().squeeze().numpy(),
    #                label='Estimate')
    # axs[1][0].plot(test_x[:, 0].detach().squeeze().numpy(),
    #             test_y_prime[:, 0,0].detach().squeeze().numpy(), label='True Y')
    # axs[1][0].legend(loc='upper right')
    # axs[1][0].fill_between(test_x[:, 0].detach().squeeze().numpy(),
    #                     lower1_p.detach().numpy(),
    #                     upper1_p.detach().numpy(), alpha=0.5)
    # axs[1][0].plot(train_X[:, 0].squeeze().detach().numpy(),
    #             train_Y_prime[:, 0, 0].detach().numpy(), 'o',
    #             color='black',
    #             markersize=5,
    #             fillstyle="none")
    # axs[1][0].title.set_text('$\\cos(\\frac{x_1}{10})$')
    # axs[1][1].plot(test_x[:, 1].detach().squeeze().numpy(),
    #                y_hat_prime[:, 1, 1].detach().squeeze().numpy(),
    #                label='Estimate')
    # axs[1][1].plot(test_x[:, 1].detach().squeeze().numpy(),
    #                test_y_prime[:, 1,1].detach().squeeze().numpy(),
    #                label='True Y')
    # axs[1][1].legend(loc='upper right')
    # axs[1][1].fill_between(test_x[:, 1].detach().squeeze().numpy(),
    #                     lower2_p.detach().numpy(),
    #                     upper2_p.detach().numpy(), alpha=0.5)
    # axs[1][1].title.set_text('$10 \\cos(\\frac{x_1}{10})$')
    # plt.show()


def test_2d2dgp_scaling_wrapping():
    #diagonals should be derivative
    # off-diagonals should be approximately zero
    sigma = 0.5
    train_X = th.stack([th.distributions.Uniform(0, 2 * pi * 10).sample((250,)),
                        th.distributions.Uniform(0, 2 * pi * 100).sample((250,))], dim=1)
    train_Y = th.stack([
        10 * th.sin(train_X[:, 0]/10),
        100 * th.cos(train_X[:,1]/ 100),
        train_X[:, 0] * train_X[:,1]
        ], dim=1) + th.distributions.Normal(
        0.0, sigma).sample((train_X.shape[0], 3))

    train_Y_prime = th.stack([
        th.stack([th.cos(train_X[:, 0] / 10), th.zeros(train_X.shape[0],)], dim=1),
        th.stack([th.zeros(train_X.shape[0],), -th.sin(train_X[:, 1] / 100)],dim=1),
        th.stack([train_X[:, 1], train_X[:, 0]], dim=1)], dim=1)

    # kernel = RBFKernel(2, ard_num_dims=True)
    kernel = AdditiveKernel(kernels=[
        PeriodicKernel(p_prior=2.),
        PeriodicKernel(p_prior=2.)],
        active_dims=[
            [0],
            [1]])
    x_scaler = GPScaler(xmins=th.tensor([0, 0]), xmaxs=th.tensor([2*pi*10, 2*pi*100]))
    y_scaler = GPScaler(xmins=th.tensor([-10, -100, 0]),
                        xmaxs=th.tensor([10,100, 4 * pi * pi * 1000]))
    gp = ScaledGaussianProcess(train_X, train_Y, kernel,
                               x_scaler=x_scaler, y_scaler=y_scaler)
    gp.train_model()
    # X, Y = th.meshgrid(dim_sample, dim_sample)
    x_vec = th.linspace(-4 * pi * 10, 4 * pi * 10, 100)
    y_vec = th.linspace(-4 * pi * 100, 4 * pi * 100, 100)
    test_x = th.stack((x_vec, y_vec), dim=1)
    test_y = th.stack([
        10 * th.sin(test_x[:,0]/10),
        100 * th.cos(test_x[:,1] / 100),
        test_x[:,0] * test_x[:,1]
    ], dim=1)

    test_y_prime = th.stack([
        th.stack([th.cos(test_x[:, 0] / 10), th.zeros(test_x.shape[0], )],
                 dim=1),
        th.stack([th.zeros(test_x.shape[0], ), -th.sin(test_x[:, 1] / 100)],
                 dim=1),
        th.stack([test_x[:, 1], test_x[:, 0]], dim=1)], dim=1)

    y_hat, cov = gp(test_x)
    y_hat_prime, cov_prime = gp.ddx(test_x)
    out_dist_1 = gpt.distributions.MultivariateNormal(y_hat.squeeze()[:,0],
                                                      cov[:,:,0,0])
    out_dist_2 = gpt.distributions.MultivariateNormal(y_hat.squeeze()[:, 1],
                                                      cov[:,:,1,1])
    out_dist_3 = gpt.distributions.MultivariateNormal(y_hat.squeeze()[:, 2],
                                                      cov[:,:,2,2])

    lower1, upper1 = out_dist_1.confidence_region()
    lower2, upper2 = out_dist_2.confidence_region()
    lower3, upper3 = out_dist_3.confidence_region()
    lower1_p , upper1_p = gpt.distributions.MultivariateNormal(
        y_hat_prime[:,0,0], cov_prime[:,:,0,0,0,0]).confidence_region()
    lower2_p , upper2_p = gpt.distributions.MultivariateNormal(
        y_hat_prime[:,1,1], cov_prime[:,:,1,1,1,1]).confidence_region()
    lower3_p, upper3_p = gpt.distributions.MultivariateNormal(
        y_hat_prime[:, 2, 1], cov_prime[:, :, 1, 1, 2,2]).confidence_region()
    # these dont' guarantee true value is in confidence region but its good
    # enough
    # GP tests
    assert (test_y[:,0] > lower1).all()
    assert (test_y[:,1] > lower2).all()
    assert (test_y[:, 0] < upper1).all()
    assert (test_y[:, 1] < upper2).all()
    assert (y_hat[:,[0,1]] - test_y[:, [0,1]]).abs().mean() < 1 #notice the
    # loss of a significant figure
    # # # derivative of GP tests
    assert abs(y_hat_prime[:, 0, 0] - test_y_prime[:, 0, 0]).mean() < 1e-1
    assert abs(y_hat_prime[:, 1, 1] - test_y_prime[:, 1, 1]).mean() < 1e-1
    assert y_hat_prime[:, 0, 1].abs().mean() < 1 #notice the loss of a significant figure
    assert y_hat_prime[:, 1, 0].abs().mean() < 1 #notice the loss of a significant figure

    assert (y_hat_prime[:, 0, 0] < upper1_p).all()
    assert (y_hat_prime[:, 0, 0] > lower1_p).all()
    assert (y_hat_prime[:, 1, 1] < upper2_p).all()
    assert (y_hat_prime[:, 1, 1] > lower2_p).all()

    #uncomment plots for debugging
    # f, axs = plt.subplots(2,3, figsize=(18, 6))
    # axs[0][0].plot(test_x[:,0].detach().squeeze().numpy(),
    #             y_hat[:,0].detach().squeeze().numpy(), label='Estimate')
    # axs[0][0].plot(test_x[:,0].detach().squeeze().numpy(),
    #             test_y[:,0].detach().squeeze().numpy(), label='True Y')
    # axs[0][0].legend(loc='upper right')
    # axs[0][0].plot(train_X[:,0].squeeze().detach().numpy(),
    #             train_Y[:,0].detach().numpy(), 'o',
    #         color='black',
    #         markersize=5,
    #         fillstyle="none")
    # axs[0][0].fill_between(test_x[:,0].detach().squeeze().numpy(),
    #                     lower1.detach().numpy(),
    #                     upper1.detach().numpy(), alpha=0.5)
    # axs[0][1].plot(test_x[:,1].detach().squeeze().numpy(),
    #             y_hat[:,1].detach().squeeze().numpy(), label='Estimate')
    # axs[0][1].plot(test_x[:, 1].detach().squeeze().numpy(),
    #             test_y[:, 1].detach().squeeze().numpy(), label='True Y')
    # axs[0][1].legend(loc='upper right')
    # axs[0][1].fill_between(test_x[:, 1].detach().squeeze().numpy(),
    #                     lower2.detach().numpy(),
    #                     upper2.detach().numpy(), alpha=0.5)
    # axs[0][1].plot(train_X[:,1].squeeze().detach().numpy(),
    #             train_Y[:,1].detach().numpy(), 'o',
    #         color='black',
    #         markersize=5,
    #         fillstyle="none")
    #
    # axs[0][2].plot(test_x[:, 1].detach().squeeze().numpy(),
    #                y_hat[:, 2].detach().squeeze().numpy(), label='Estimate')
    # axs[0][2].plot(test_x[:, 1].detach().squeeze().numpy(),
    #                test_y[:, 2].detach().squeeze().numpy(), label='True Y')
    # axs[0][2].legend(loc='upper right')
    # axs[0][2].fill_between(test_x[:, 1].detach().squeeze().numpy(),
    #                        lower3.detach().numpy(),
    #                        upper3.detach().numpy(), alpha=0.5)
    # axs[0][2].plot(train_X[:, 1].squeeze().detach().numpy(),
    #                train_Y[:, 2].detach().numpy(), 'o',
    #                color='black',
    #                markersize=5,
    #                fillstyle="none")
    #
    # axs[1][0].plot(test_x[:, 0].detach().squeeze().numpy(),
    #             y_hat_prime[:, 0,0].detach().squeeze().numpy(),
    #                label='Estimate')
    # axs[1][0].plot(test_x[:, 0].detach().squeeze().numpy(),
    #             test_y_prime[:, 0, 0].detach().squeeze().numpy(), label='True Y')
    # axs[1][0].legend(loc='upper right')
    # axs[1][0].fill_between(test_x[:, 0].detach().squeeze().numpy(),
    #                     lower1_p.detach().numpy(),
    #                     upper1_p.detach().numpy(), alpha=0.5)
    # axs[1][0].plot(train_X[:, 0].squeeze().detach().numpy(),
    #             train_Y_prime[:, 0, 0].detach().numpy(), 'o',
    #             color='black',
    #             markersize=5,
    #             fillstyle="none")
    #
    # axs[1][1].plot(test_x[:, 1].detach().squeeze().numpy(),
    #                y_hat_prime[:, 1, 1].detach().squeeze().numpy(),
    #                label='Estimate')
    # axs[1][1].plot(test_x[:, 1].detach().squeeze().numpy(),
    #                test_y_prime[:, 1, 1].detach().squeeze().numpy(),
    #                label='True Y')
    # axs[1][1].legend(loc='upper right')
    # axs[1][1].fill_between(test_x[:, 1].detach().squeeze().numpy(),
    #                     lower2_p.detach().numpy(),
    #                     upper2_p.detach().numpy(), alpha=0.5)
    # axs[1][1].plot(train_X[:, 1].squeeze().detach().numpy(),
    #                train_Y_prime[:, 1, 1].detach().numpy(), 'o',
    #                color='black',
    #                markersize=5,
    #                fillstyle="none")
    #
    # axs[1][2].plot(test_x[:, 1].detach().squeeze().numpy(),
    #                y_hat_prime[:, 2, 1].detach().squeeze().numpy(),
    #                label='Estimate')
    # test_y_prime.shape
    # axs[1][2].plot(test_x[:, 1].detach().squeeze().numpy(),
    #                test_y_prime[:, 2,1].detach().squeeze().numpy(),
    #                label='True Y')
    # axs[1][2].legend(loc='upper right')
    # axs[1][2].fill_between(test_x[:, 1].detach().squeeze().numpy(),
    #                     lower3_p.detach().numpy(),
    #                     upper3_p.detach().numpy(), alpha=0.5)
    # axs[1][2].plot(train_X[:, 1].squeeze().detach().numpy(),
    #                train_Y_prime[:, 2, 1].detach().numpy(), 'o',
    #                color='black',
    #                markersize=5,
    #                fillstyle="none")
    # plt.show()

def test_2d2dgp_scaling():
    #diagonals should be derivative
    # off-diagonals should be approximately zero

    sigma = 0.5
    train_X = th.stack([th.distributions.Uniform(0, 2 * pi * 10).sample((250,)),
                        th.distributions.Uniform(0, 2 * pi * 100).sample((250,))], dim=1)
    train_Y = th.stack([
        10 * th.sin(train_X[:, 0]/10),
        100 * th.cos(train_X[:,1]/ 100)], dim=1) + th.distributions.Normal(0.0, sigma).sample(train_X.shape)

    train_Y_prime = th.stack([
        th.cat([th.cos(train_X[:, 0]/10).unsqueeze(1), th.zeros(train_X.shape[0], 1)],dim=1),
        th.cat([th.zeros(train_X.shape[0], 1), -th.sin(train_X[:,1]/100).unsqueeze(1)],
               dim=1)],
        dim=1)
    kernel = RBFKernel(2, ard_num_dims=True)
    x_scaler = GPScaler(xmins=th.tensor([0, 0]), xmaxs=th.tensor([2*pi*10, 2*pi*100]))
    y_scaler = GPScaler(xmins=th.tensor([-10, -100]) , xmaxs=th.tensor([10, 100]))
    gp = ScaledGaussianProcess(train_X, train_Y, kernel,
                               x_scaler=x_scaler, y_scaler=y_scaler)
    gp.train_model()
    # X, Y = th.meshgrid(dim_sample, dim_sample)
    x_vec = th.linspace(0, 2 * pi * 10, 100)
    y_vec = th.linspace(0, 2 * pi * 100, 100)
    test_x = th.stack((x_vec, y_vec), dim=1)
    test_y = th.stack([
        10 * th.sin(test_x[:,0]/10),
        100 * th.cos(test_x[:,1] / 100)], dim=1)

    test_y_prime = th.stack([
        th.cos(test_x[:,0]/10),
        -th.sin(test_x[:,1]/100)], dim=1)

    y_hat, cov = gp(test_x)
    y_hat_prime, cov_prime = gp.ddx(test_x)
    out_dist_1 = gpt.distributions.MultivariateNormal(y_hat.squeeze()[:,0],
                                                      cov[:,:,0,0])
    out_dist_2 = gpt.distributions.MultivariateNormal(y_hat.squeeze()[:, 1],
                                                      cov[:,:,1,1])
    lower1, upper1 = out_dist_1.confidence_region()
    lower2, upper2 = out_dist_2.confidence_region()
    lower1_p , upper1_p = gpt.distributions.MultivariateNormal(
        y_hat_prime[:,0,0], cov_prime[:,:,0,0, 0,0]).confidence_region()
    lower2_p , upper2_p = gpt.distributions.MultivariateNormal(
        y_hat_prime[:,1,1], cov_prime[:,:,1,1, 1, 1]).confidence_region()

    # these dont' guarantee true value is in confidence region but its good
    # enough
    # GP tests
    assert (test_y[:,0] > lower1).all()
    assert (test_y[:,1] > lower2).all()
    assert (test_y[:, 0] < upper1).all()
    assert (test_y[:, 1] < upper2).all()
    assert (y_hat - test_y).abs().mean() < 1 #notice the loss of a significant figure
    # # derivative of GP tests
    # assert abs(y_hat_prime[:, 0, 0] - test_y_prime[:, 0]).mean() < 1e-1
    # assert abs(y_hat_prime[:, 1, 1] - test_y_prime[:, 1]).mean() < 1e-1
    assert y_hat_prime[:, 0, 1].abs().mean() < 1 #notice the loss of a significant figure
    assert y_hat_prime[:, 1, 0].abs().mean() < 1 #notice the loss of a significant figure

    assert (y_hat_prime[:, 0, 0] < upper1_p).all()
    assert (y_hat_prime[:, 0, 0] > lower1_p).all()
    assert (y_hat_prime[:, 1, 1] < upper2_p).all()
    assert (y_hat_prime[:, 1, 1] > lower2_p).all()

    #uncomment plots for debugging
    # f, axs = plt.subplots(2,2)
    # axs[0][0].plot(test_x[:,0].detach().squeeze().numpy(),
    #             y_hat[:,0].detach().squeeze().numpy(), label='Estimate')
    # axs[0][0].plot(test_x[:,0].detach().squeeze().numpy(),
    #             test_y[:,0].detach().squeeze().numpy(), label='True Y')
    # axs[0][0].legend(loc='upper right')
    # axs[0][0].plot(train_X[:,0].squeeze().detach().numpy(),
    #             train_Y[:,0].detach().numpy(), 'o',
    #         color='black',
    #         markersize=5,
    #         fillstyle="none")
    # axs[0][0].fill_between(test_x[:,0].detach().squeeze().numpy(),
    #                     lower1.detach().numpy(),
    #                     upper1.detach().numpy(), alpha=0.5)
    # axs[0][1].plot(test_x[:,1].detach().squeeze().numpy(),
    #             y_hat[:,1].detach().squeeze().numpy(), label='Estimate')
    # axs[0][1].plot(test_x[:, 1].detach().squeeze().numpy(),
    #             test_y[:, 1].detach().squeeze().numpy(), label='True Y')
    # axs[0][1].legend(loc='upper right')
    # axs[0][1].fill_between(test_x[:, 1].detach().squeeze().numpy(),
    #                     lower2.detach().numpy(),
    #                     upper2.detach().numpy(), alpha=0.5)
    # axs[0][1].plot(train_X[:,1].squeeze().detach().numpy(),
    #             train_Y[:,1].detach().numpy(), 'o',
    #         color='black',
    #         markersize=5,
    #         fillstyle="none")
    #
    # axs[1][0].plot(test_x[:, 0].detach().squeeze().numpy(),
    #             y_hat_prime[:, 0,0].detach().squeeze().numpy(),
    #                label='Estimate')
    # axs[1][0].plot(test_x[:, 0].detach().squeeze().numpy(),
    #             test_y_prime[:, 0].detach().squeeze().numpy(), label='True Y')
    # axs[1][0].legend(loc='upper right')
    # axs[1][0].fill_between(test_x[:, 0].detach().squeeze().numpy(),
    #                     lower1_p.detach().numpy(),
    #                     upper1_p.detach().numpy(), alpha=0.5)
    # axs[1][0].plot(train_X[:, 0].squeeze().detach().numpy(),
    #             train_Y_prime[:, 0, 0].detach().numpy(), 'o',
    #             color='black',
    #             markersize=5,
    #             fillstyle="none")
    #
    # axs[1][1].plot(test_x[:, 1].detach().squeeze().numpy(),
    #                y_hat_prime[:, 1, 1].detach().squeeze().numpy(),
    #                label='Estimate')
    # axs[1][1].plot(test_x[:, 1].detach().squeeze().numpy(),
    #                test_y_prime[:, 1].detach().squeeze().numpy(),
    #                label='True Y')
    # axs[1][1].legend(loc='upper right')
    # axs[1][1].fill_between(test_x[:, 1].detach().squeeze().numpy(),
    #                     lower2_p.detach().numpy(),
    #                     upper2_p.detach().numpy(), alpha=0.5)
    # axs[1][1].plot(train_X[:, 1].squeeze().detach().numpy(),
    #                train_Y_prime[:, 1, 1].detach().numpy(), 'o',
    #                color='black',
    #                markersize=5,
    #                fillstyle="none")
    # plt.show()

def test_2d2dgp():
    #diagonals should be derivative
    # off-diagonals should be approximately zero
    sigma = 0.2
    train_X = th.rand((250,2))*2*pi
    train_Y = th.stack([
        th.sin(train_X[:, 0]),
        th.cos(train_X[:,1])],dim=1) + sigma * (th.rand_like(train_X) - 0.5)

    train_Y_prime = th.stack([
        th.cat([th.cos(train_X[:, 0]).unsqueeze(1), th.zeros(train_X.shape[0], 1)],dim=1),
        th.cat([th.zeros(train_X.shape[0], 1), -th.sin(train_X[:, 1]).unsqueeze(1)],dim=1)],
        dim=1)
    kernel = RBFKernel(2)
    gp = GaussianProcess(train_X, train_Y, kernel)
    gp.train_model()
    dim_sample = th.linspace(0, 2 * pi, 100)
    # X, Y = th.meshgrid(dim_sample, dim_sample)
    x_vec, y_vec = dim_sample, dim_sample
    test_x = th.stack((x_vec, y_vec), dim=1)
    test_y = th.stack([
        th.sin(test_x[:,0]),
        th.cos(test_x[:,1])], dim=1)

    test_y_prime = th.stack([
        th.cos(test_x[:,0]),
        -th.sin(test_x[:,1])], dim=1)

    y_hat, cov = gp(test_x)
    cov = cov.to_dense()
    y_hat_prime, cov_prime = gp.ddx(test_x)
    cov_prime = cov_prime.to_dense()
    out_dist_1 = gpt.distributions.MultivariateNormal(y_hat.squeeze()[:,0],
                                                      cov[:,:,0, 0])
    out_dist_2 = gpt.distributions.MultivariateNormal(y_hat.squeeze()[:, 1],
                                                      cov[:,:,1, 1])
    lower1, upper1 = out_dist_1.confidence_region()
    lower2, upper2 = out_dist_2.confidence_region()
    lower1_p , upper1_p = gpt.distributions.MultivariateNormal(
        y_hat_prime[:,0,0], cov_prime[:,:,0,0, 0, 0]).confidence_region()
    lower2_p , upper2_p = gpt.distributions.MultivariateNormal(
        y_hat_prime[:,1,1], cov_prime[:,:,1,1, 1, 1]).confidence_region()

    # these dont' guarantee true value is in confidence region but its good
    # enough
    # GP tests
    assert (test_y[:,0] > lower1).all()
    assert (test_y[:,1] > lower2).all()
    assert (test_y[:, 0] < upper1).all()
    assert (test_y[:, 1] < upper2).all()
    assert (y_hat - test_y).abs().mean() < 1e-1
    # # derivative of GP tests
    assert abs(y_hat_prime[:, 0, 0] - test_y_prime[:, 0]).mean() < 1e-1
    assert abs(y_hat_prime[:, 1, 1] - test_y_prime[:, 1]).mean() < 1e-1
    assert y_hat_prime[:, 0, 1].abs().mean() < 1e-1
    assert y_hat_prime[:, 1, 0].abs().mean() < 1e-1
    #
    assert (y_hat_prime[:, 0, 0] < upper1_p).all()
    assert (y_hat_prime[:, 0, 0] > lower1_p).all()
    assert (y_hat_prime[:, 1, 1] < upper2_p).all()
    assert (y_hat_prime[:, 1, 1] > lower2_p).all()

    #uncomment plots for debugging
    # f, axs = plt.subplots(2,2)
    # axs[0][0].plot(test_x[:,0].detach().squeeze().numpy(),
    #             y_hat[:,0].detach().squeeze().numpy(), label='Estimate')
    # axs[0][0].plot(test_x[:,0].detach().squeeze().numpy(),
    #             test_y[:,0].detach().squeeze().numpy(), label='True Y')
    # axs[0][0].legend(loc='upper right')
    # axs[0][0].plot(train_X[:,0].squeeze().detach().numpy(),
    #             train_Y[:,0].detach().numpy(), 'o',
    #         color='black',
    #         markersize=5,
    #         fillstyle="none")
    # axs[0][0].fill_between(test_x[:,0].detach().squeeze().numpy(),
    #                     lower1.detach().numpy(),
    #                     upper1.detach().numpy(), alpha=0.5)
    # axs[0][1].plot(test_x[:,1].detach().squeeze().numpy(),
    #             y_hat[:,1].detach().squeeze().numpy(), label='Estimate')
    # axs[0][1].plot(test_x[:, 1].detach().squeeze().numpy(),
    #             test_y[:, 1].detach().squeeze().numpy(), label='True Y')
    # axs[0][1].legend(loc='upper right')
    # axs[0][1].fill_between(test_x[:, 1].detach().squeeze().numpy(),
    #                     lower2.detach().numpy(),
    #                     upper2.detach().numpy(), alpha=0.5)
    # axs[0][1].plot(train_X[:,1].squeeze().detach().numpy(),
    #             train_Y[:,1].detach().numpy(), 'o',
    #         color='black',
    #         markersize=5,
    #         fillstyle="none")
    #
    # axs[1][0].plot(test_x[:, 0].detach().squeeze().numpy(),
    #             y_hat_prime[:, 0,0].detach().squeeze().numpy(),
    #                label='Estimate')
    # axs[1][0].plot(test_x[:, 0].detach().squeeze().numpy(),
    #             test_y_prime[:, 0].detach().squeeze().numpy(), label='True Y')
    # axs[1][0].legend(loc='upper right')
    # axs[1][0].fill_between(test_x[:, 0].detach().squeeze().numpy(),
    #                     lower1_p.detach().numpy(),
    #                     upper1_p.detach().numpy(), alpha=0.5)
    # axs[1][0].plot(train_X[:, 0].squeeze().detach().numpy(),
    #             train_Y_prime[:, 0, 0].detach().numpy(), 'o',
    #             color='black',
    #             markersize=5,
    #             fillstyle="none")
    #
    # axs[1][1].plot(test_x[:, 1].detach().squeeze().numpy(),
    #                y_hat_prime[:, 1, 1].detach().squeeze().numpy(),
    #                label='Estimate')
    # axs[1][1].plot(test_x[:, 1].detach().squeeze().numpy(),
    #                test_y_prime[:, 1].detach().squeeze().numpy(),
    #                label='True Y')
    # axs[1][1].legend(loc='upper right')
    # axs[1][1].fill_between(test_x[:, 1].detach().squeeze().numpy(),
    #                     lower2_p.detach().numpy(),
    #                     upper2_p.detach().numpy(), alpha=0.5)
    # axs[1][1].plot(train_X[:, 1].squeeze().detach().numpy(),
    #                train_Y_prime[:, 1, 1].detach().numpy(), 'o',
    #                color='black',
    #                markersize=5,
    #                fillstyle="none")
    # plt.show()


def test_exp_kernel():
    test_X = th.tensor([[7.1,-100.2], [0.5, 12.2]], dtype=th.float64)
    kernel = RBFKernel(2)
    kernel._length_scale.data = th.tensor(1.23)
    kernel._signal_variance.data = th.tensor(4.56)

    K = kernel(test_X, test_X)
    dKdx1 = kernel.ddx1(test_X, test_X)
    dKdx2 = kernel.ddx2(test_X, test_X)
    d2Kdx1x2 = kernel.d2dx1x2(test_X, test_X)

    K_expected = th.tensor(
        [[95.5835, 6.18301e-234],
         [6.18301e-234, 95.5835]])
    dKdx1_expected = th.tensor([[[0, 0], [-3.48642e-234, 5.93748e-233]], [[3.48642e-234, -5.93748e-233], [0, 0]]])
    dKdx2_expected = th.tensor([[[0, 0], [3.48642e-234, -5.93748e-233]], [[-3.48642e-234,5.93748e-233], [0, 0]]])

    d2Kdx1x2_expected = th.tensor([[[[8.166169912567646, 0],
                                     [0, 8.166169912567646]],
                           [[-1.43764e-234,3.34797e-233], [3.34797e-233, -5.69641e-232]]], \
                           [[[-1.43764e-234, 3.34797e-233], [3.34797e-233, -5.69641e-232]],
                            [[8.166169912567646,0],
                             [0, 8.166169912567646]]]])

    th.testing.assert_allclose(K, K_expected)
    th.testing.assert_allclose(dKdx1, dKdx1_expected, atol=1e-234, rtol=1e-234)
    th.testing.assert_allclose(dKdx2, dKdx2_expected, atol=1e-234, rtol=1e-234)
    #Diagonals of first and alst elements are a bit weird.
    #probably because we are doing a partial not full derivative
    #even when x1 == x2. Mathematica shortcuts to zero in these cases.
    th.testing.assert_allclose(d2Kdx1x2, d2Kdx1x2_expected, atol=1e-234, rtol=1e-234)


def test_periodic_kernel():
    test_X = th.tensor([[7.1,-100.2], [0.5, 12.2]], dtype=th.float64)
    kernel = PeriodicKernel(p_prior=3)
    kernel._length_scale.data = th.tensor(1.23)
    kernel._signal_variance.data = th.tensor(4.56)

    K = kernel(test_X[:, 0, None], test_X[:, 0, None])
    dKdx1 = kernel.ddx1(test_X[:,0,None], test_X[:,0,None])
    dKdx2 = kernel.ddx2(test_X[:,0,None], test_X[:,0,None])
    d2Kdx1x2 = kernel.d2dx1x2(test_X[:,0,None], test_X[:,0,None])

    K_expected = th.tensor([[95.5835, 90.1041], [90.1041, 95.5835]])
    dKdx1_expected = th.tensor([[0, -15.3336], [15.3336, 0]])
    dKdx2_expected = th.tensor([[0, 15.3336], [-15.3336, 0]])
    d2Kdx1x2_expected = th.tensor([[35.8208, 7.82527], [7.82527, 35.8208]])

    th.testing.assert_allclose(K, K_expected, atol=1e-6, rtol=1e-6)
    th.testing.assert_allclose(dKdx1, dKdx1_expected, atol=1e-5, rtol=1e-5)
    th.testing.assert_allclose(dKdx2, dKdx2_expected, atol=1e-5, rtol=1e-5)
    th.testing.assert_allclose(d2Kdx1x2, d2Kdx1x2_expected, atol=1e-6, rtol=1e-6)

def test_additive_kernel():
    test_X = th.tensor([[7.1, -100.2], [0.5, 12.2]], dtype=th.float64)
    p_kernel = PeriodicKernel(p_prior=3)
    rbf_kernel = RBFKernel(2)
    p_kernel._length_scale.data = th.tensor(1.23)
    p_kernel._signal_variance.data = th.tensor(4.56)
    rbf_kernel._length_scale.data = th.tensor(1.23)
    rbf_kernel._signal_variance.data = th.tensor(4.56)
    kernel = AdditiveKernel(kernels=[p_kernel, rbf_kernel],
                            active_dims=[[0], [0,1]])
    K = kernel(test_X)
    dKdx1 = kernel.ddx1(test_X, test_X)
    dKdx2 = kernel.ddx2(test_X, test_X)
    d2Kdx1x2 = kernel.d2dx1x2(test_X, test_X)

    rbf_K_expected = th.tensor(
        [[95.5835, 6.18301e-234],
         [6.18301e-234, 95.5835]])
    rbf_dKdx1_expected = th.tensor([[[0, 0], [-3.48642e-234, 5.93748e-233]],
                                [[3.48642e-234, -5.93748e-233], [0, 0]]])
    rbf_dKdx2_expected = th.tensor([[[0, 0], [3.48642e-234, -5.93748e-233]],
                                [[-3.48642e-234, 5.93748e-233], [0, 0]]])

    rbf_d2Kdx1x2_expected = th.tensor([[[[8.166169912567646, 0],
                                     [0, 8.166169912567646]],
                                    [[-1.43764e-234, 3.34797e-233],
                                     [3.34797e-233, -5.69641e-232]]], \
                                   [[[-1.43764e-234, 3.34797e-233],
                                     [3.34797e-233, -5.69641e-232]],
                                    [[8.166169912567646, 0],
                                     [0, 8.166169912567646]]]])
    p_K_expected = th.tensor([[95.5835, 90.1041], [90.1041, 95.5835]])
    p_dKdx1_expected = th.zeros_like(rbf_dKdx1_expected)
    p_dKdx1_expected[:, :,0] = th.tensor([[0, -15.3336], [15.3336, 0]])
    p_dKdx2_expected = th.zeros_like(rbf_dKdx1_expected)
    p_dKdx2_expected[:, :, 0] = th.tensor([[0, 15.3336], [-15.3336, 0]])
    p_d2Kdx1x2_expected = th.zeros_like(rbf_d2Kdx1x2_expected)
    p_d2Kdx1x2_expected[:,:,0,0] = th.tensor([[35.8208, 7.82527], [7.82527, 35.8208]])

    K_expected = rbf_K_expected + p_K_expected
    dKdx1_expected = rbf_dKdx1_expected + p_dKdx1_expected
    dKdx2_expected = rbf_dKdx2_expected + p_dKdx2_expected
    d2Kdx1x2_expected = rbf_d2Kdx1x2_expected + p_d2Kdx1x2_expected

    th.testing.assert_allclose(K, K_expected)
    th.testing.assert_allclose(dKdx1, dKdx1_expected, atol=1e-5, rtol=1e-5)
    th.testing.assert_allclose(dKdx2, dKdx2_expected, atol=1e-5, rtol=1e-5)
    th.testing.assert_allclose(d2Kdx1x2, d2Kdx1x2_expected, atol=1e-234, rtol=1e-5)

def test_multiplicative_kernel():
    test_X = th.tensor([[7.1, -100.2], [0.5, 12.2]], dtype=th.float64)
    p_kernel = PeriodicKernel(p_prior=3)
    rbf_kernel = RBFKernel(1)
    p_kernel._length_scale.data = th.tensor(1.23)
    p_kernel._signal_variance.data = th.tensor(4.56)
    rbf_kernel._length_scale.data = th.tensor(1.23)
    rbf_kernel._signal_variance.data = th.tensor(4.56)
    kernel = MultiplicativeKernel(kernels=[p_kernel, rbf_kernel],
                            active_dims=[[0], [1]])
    K = kernel(test_X, test_X)
    dKdx1 = kernel.ddx1(test_X, test_X)
    dKdx2 = kernel.ddx2(test_X, test_X)
    d2Kdx1x2 = kernel.d2dx1x2(test_X, test_X)

    K_expected = th.tensor([[9136.2, 3.58153e-231], [3.58153e-231, 9136.2]])
    dKdx1_expected = th.tensor([[[0., 0.], [-6.09493e-232,
   3.4393e-230]], [[6.09493e-232, -3.4393e-230], [0., 0.]]])
    dKdx2_expected = th.tensor([[[0., 0.], [6.09493e-232, -3.4393e-230]], [[-6.09493e-232,
   3.4393e-230], [0., 0.]]])
    d2Kdx1x2_expected = th.tensor([[[[3423.88, 0.], [0., 780.551]], [[3.11045e-232, 
    5.85289e-231], [5.85289e-231, -3.29966e-229]]], \
[[[3.11045e-232, 
    5.85289e-231], [5.85289e-231, -3.29966e-229]], [[3423.88,
     0.], [0., 780.551]]]])
    th.testing.assert_allclose(K, K_expected)
    th.testing.assert_allclose(dKdx1, dKdx1_expected, atol=1e-234, rtol=1e-5)
    th.testing.assert_allclose(dKdx2, dKdx2_expected, atol=1e-234, rtol=1e-5)
    th.testing.assert_allclose(d2Kdx1x2, d2Kdx1x2_expected, atol=1e-234, rtol=1e-5)

def test_affine_dot_product_kernel():
    test_X = th.tensor([[7.1, -100.2, -1., 1.3],
                        [0.5, 12.2, 3, 6.7]],
                       dtype=th.float64)
    p_kernel = PeriodicKernel(p_prior=3)
    rbf_kernel = RBFKernel(1)
    p_kernel._length_scale.data = th.tensor(1.23)
    p_kernel._signal_variance.data = th.tensor(4.56)
    rbf_kernel._length_scale.data = th.tensor(1.23)
    rbf_kernel._signal_variance.data = th.tensor(4.56)

    sub_kernels = [MultiplicativeKernel(kernels=[p_kernel, rbf_kernel],
                                    active_dims=[[0], [1]])]*3
    kernel = AffineDotProductKernel(s_idx=[0,1], m_idx=[2,3],
                           kernels=sub_kernels, last_is_unit=True)
    K = kernel(test_X, test_X)
    dKdx1 = kernel.ddx1(test_X, test_X)
    dKdx2 = kernel.ddx2(test_X, test_X)
    d2Kdx1x2 = kernel.d2dx1x2(test_X, test_X)


def test_multiplicative_periodic_consistency():
    kernel = MultiplicativeKernel(
        kernels=[PeriodicKernel(p_prior=1/2,
                                learn_period=False),
                 RBFKernel(2, ard_num_dims=True)],
        active_dims=[[0], [1, 2]]
    )
    expected_1 = kernel.kernels[0](th.tensor([[-1.]]), th.tensor([[-1.]])) * \
    kernel.kernels[1](
        th.tensor([[0.5, 0]]), th.tensor([[0.5, 0]]))
    expected_2 = kernel.kernels[0](th.tensor([[-1.]]), th.tensor([[-2.]])) * \
    kernel.kernels[1](
        th.tensor([[0.5, 0]]), th.tensor([[0.5, 0]]))

    actual_1 = kernel(th.tensor([[-1, 0.5, 0]]), th.tensor([[-1, 0.5, 0]]))
    actual_2 = kernel(th.tensor([[-1, 0.5, 0]]), th.tensor([[-2, 0.5, 0]]))
    actual_3 = kernel(th.tensor([[1, 0.5, 0]]), th.tensor([[1, 0.5, 0]]))
    actual_4 = kernel(th.tensor([[1, 0.5, 0]]), th.tensor([[2, 0.5, 0]]))
    th.testing.assert_allclose(expected_1, expected_2)
    th.testing.assert_allclose(actual_1, expected_1)
    th.testing.assert_allclose(actual_2, expected_2)
    th.testing.assert_allclose(actual_3, actual_4)
    th.testing.assert_allclose(actual_3, actual_2)

