import pytest
from numpy import allclose, linspace, ones, load
from core.systems import CartPole, InvertedPendulum, DoubleInvertedPendulum
import core.controllers as ctrl
from pathlib import Path
import torch as th
import numpy as np
import matplotlib.pyplot as plt

# Physical and other constants
m = 0.25
l = 0.5
T = 5
dt = 1e-2
f = 1 / dt
N = int(T * f)
ts = linspace(0, T, N + 1)


def constant_controller(system):
    return ctrl.ConstantController(system, th.ones((1,system.m)) * 0.01)


destination_dir = Path(__file__).parent.absolute() / 'data' / 'test_systems'


# Make sure system dynamics have not changed
# systems and initial conditions

@pytest.mark.parametrize('system',
                         [
                             InvertedPendulum(mass=m, l=l),
                             CartPole(m_c=m * 0.5, m_p=m * .25, l=l * 0.5),
                             DoubleInvertedPendulum(m_1=m, m_2=m * 2, l_1=l,
                                                    l_2=l * 2)
                         ])
def test_with_constrant_controller(system):
    load_dir = destination_dir / system.__class__.__name__
    ts = load(load_dir / 'ts.npy')
    true_xs = load(load_dir / 'xs.npy')
    x_0 = load(load_dir / 'x_0.npy')
    true_us = load(load_dir / 'us.npy')

    (actual_xs, actual_us) = system.simulate(x_0=th.from_numpy(x_0[None]),
                                             controller=constant_controller(system),
                                             ts=ts)
    # TODO: Should tolerance be lower? The more chaotic systems need it.

    # I suspect torch.solve is the cause
    assert allclose(actual_xs.detach().numpy(), true_xs, atol=1e-5)
    assert allclose(actual_us.detach().numpy(), true_us, atol=1e-5)

def test_cartpole():
    system = CartPole(m_c=1, m_p=2, l=3)
    ts = th.linspace(0, 10, 1000)
    x_0 = th.tensor([
     [0.0, 0.0, 0.0, 0.0],
     [3.0, 0.0, -1.0, 0.0],
     [3.0, 0.0, 1.0, 0.0],
     [-3.0, 0.0, 1.0, 0.0],
     [-3.0, 0.0, -1.0, 0.0],
     [0.0, -np.pi, 0.0, 0.0],
     [0.0, np.pi, 0.0, 1.0],
     [0.0, np.pi, 0.0, -1.0],
     [0.0, 0.0, 0.0, 1.0],
     [0.0, 0.0, 0.0, -1.0],
    ])
    controller = ctrl.ConstantController(system, th.ones((x_0.shape[0],
                                                          system.m)) * -1.0)
    (actual_xs, actual_us) = system.simulate(x_0=x_0,
                                             controller=controller,
                                             ts=ts)
    actual_xs = actual_xs.detach().numpy()
    actual_us = actual_us.detach().numpy()
    # for i in range(actual_xs.shape[0]):
    #     system.plot(actual_xs[i], actual_us[i], ts)
    #     plt.show()
    print(actual_xs.shape)
    print(actual_us.shape)


