import pytest
from numpy import allclose, linspace, ones, load
from core.systems import CartPole, InvertedPendulum, DoubleInvertedPendulum
import core.controllers as ctrl
from pathlib import Path

# Physical and other constants
m = 0.25
l = 0.5
T = 5
dt = 1e-2
f = 1 / dt
N = int(T * f)
ts = linspace(0, T, N + 1)


def constant_controller(system):
    return ctrl.ConstantController(system, ones([system.m]) * 0.01)


destination_dir = Path(__file__).parent.absolute() / 'data' / 'test_systems'


# Make sure system dynamics have not changed
# systems and initial conditions

@pytest.mark.parametrize('system',
                         [
                             InvertedPendulum(m=m, l=l),
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
    (actual_xs, actual_us) = system.simulate(x_0=x_0,
                                             controller=constant_controller(
                                                 system),
                                             ts=ts)
    # TODO: Should tolerance be lower? The more chaotic systems need it.
    # I suspect torch.solve is the cause
    assert allclose(actual_xs, true_xs, atol=1e-4)
    assert allclose(actual_us, true_us, atol=1e-4)
