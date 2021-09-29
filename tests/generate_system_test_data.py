from numpy import array, linspace, ones, pi, save
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


def main():
    destination_dir = Path(__file__).parent.absolute() / 'data' / 'test_systems'

    # Define most systems

    systems = [InvertedPendulum(mass=m, l=l),
               CartPole(m_c=m * 0.5, m_p=m * .25, l=l * 0.5),
               DoubleInvertedPendulum(m_1=m, m_2=m * 2, l_1=l, l_2=l * 2)]

    x0s = [array([pi / 4, .01]),
           array([0, pi, .1, -.1]),
           array([0, pi, .1, -.1])]

    for sys, x_0 in zip(systems, x0s):
        (xs, us) = sys.simulate(x_0=x_0,
                                controller=constant_controller(sys), ts=ts)
        system_path = destination_dir / sys.__class__.__name__
        system_path.mkdir(exist_ok=True, parents=True)
        save(system_path / 'xs', xs)
        save(system_path / 'us', us)
        save(system_path / 'ts', ts)
        save(system_path / 'x_0', x_0)


if __name__ == '__main__':
    main()
