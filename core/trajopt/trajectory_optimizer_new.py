import cvxpy as cp
import numpy as np
import torch
from core.controllers import PiecewiseConstantController, MPCController
from core.dynamics.affine_dynamics import AffineDynamics


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""[TODO]
* terminal/traj constraints
* examples w/3d quadrotor
* obstacles
"""

class TrajectoryOptimizer:
    """A QP solver that solves the finite-horizon optimal control problem."""

    def __init__(
        self,
        dyn: AffineDynamics,
        N: int,
        dt: float,
        Q: np.ndarray,
        R: np.ndarray,
        Qf: np.ndarray,
        u_min: np.ndarray | None = None,
        u_max: np.ndarray | None = None,
        # trapezoidal: bool = False,
    ) -> None:
        """Initializes the optimizer.

        Parameters
        ----------
        dyn : AffineDynamics
            A control affine dynamical system.
        N : int
            The number of time steps including the initial state.
        dt : float
            The time discretization of the trajectory optimizer.
        Q : np.ndarray, shape=(n, n)
            The stage state cost matrix.
        R : np.ndarray, shape=(m, m)
            The stage input cost matrix.
        Qf : np.ndarray, shape=(n, n)
            The terminal cost matrix.
        u_min : np.ndarray | None, shape=(m,)
            Lower bounds on the control input.
        u_max : np.ndarray | None, shape=(m,)
            Upper bounds on the control input.

        Unused
        ------
        trapezoidal : bool, default=False
            Whether to use trapezoidal collocation.
        """
        assert isinstance(dyn, AffineDynamics)
        assert isinstance(N, int)
        assert isinstance(dt, float)
        assert isinstance(Q, np.ndarray)
        assert isinstance(R, np.ndarray)
        assert isinstance(Qf, np.ndarray)
        assert isinstance(u_min, np.ndarray) or u_min is None
        assert isinstance(u_max, np.ndarray) or u_max is None
        # assert isinstance(trapezoidal, bool)

        assert N > 1
        assert dt > 0.0
        assert Q.shape == (dyn.n, dyn.n)
        assert R.shape == (dyn.m, dyn.m)
        assert Qf.shape == (dyn.n, dyn.n)
        if u_min is not None:
            assert len(u_min.shape) == 1
        if u_max is not None:
            assert len(u_max.shape) == 1
        if u_min is not None and u_max is not None:
            assert u_min.shape == u_max.shape

        # setting variables
        self.dyn = dyn
        self.N = N
        self.dt = dt
        self.u_min = u_min
        self.u_max = u_max
        # self.trapezoidal = trapezoidal
        self.n = dyn.n  # state dimension
        self.m = dyn.m  # control dimension

        # cvxpy Variables and Parameters
        self.Q = Q
        self.R = R
        self.Qf = Qf

        self.x0 = cp.Parameter((self.n))  # initial state
        self.xs = cp.Variable((N, self.n))  # [DECISION VARIABLE] states
        self.us = cp.Variable((N - 1, self.m))  # [DECISION VARIABLE] control inputs
        
        # setting constraints
        self.Fs = [cp.Parameter((self.n, self.n)) for _ in range(N - 1)]
        self.Gs = [cp.Parameter((self.n, self.m)) for _ in range(N - 1)]
        self.Hs = [cp.Parameter((self.n)) for _ in range(N - 1)]

        self.constraints = [self.xs[0, :] == self.x0]  # initial condition
        for k in range(N - 1):
            F = self.Fs[k]
            G = self.Gs[k]
            H = self.Hs[k]
            self.constraints += [
                self.xs[k + 1, :] == F @ self.xs[k, :] + G @ self.us[k, :] + H
            ]  # dynamics constraints
            if self.u_min is not None:
                self.constraints += [
                    self.u_min <= self.us[k, :]
                ]  # minimum input constraints
            if self.u_max is not None:
                self.constraints += [
                    self.us[k, :] <= self.u_max
                ]  # maximum input constraints

        # setting cost
        self.cost = cp.quad_form(self.xs[-1, :], Qf)  # terminal cost
        for k in range(N - 1):
            self.cost += cp.quad_form(self.xs[k, :], Q)  # stage state cost
            self.cost += cp.quad_form(self.us[k, :], R)  # stage input cost

        # initializing problem
        self.prob = cp.Problem(cp.Minimize(self.cost), self.constraints)

    def _compute_FGHs(
        self, x_bars: torch.Tensor, u_bars: torch.Tensor, ts: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes the F, G, H matrices from the system dynamics at each step.

        x(t + h) = F @ x(t) + G @ u(t) + H is the exact solution to the diffeq
        dx = A @ x(t) + B @ u(t) + C
        where
        [F, G, H; 0, 0, 0; 0, 0, 0] = expm(dt * [A, B, C; 0, 0, 0; 0, 0, 0])
        assuming a ZOH control input.

        Parameters
        ----------
        x_bars : torch.Tensor, shape=(N, n)
            The state trajectory about which to linearize the dynamics.
        u_bars : torch.Tensor, shape=(N - 1, m)
            The input trajectory about which to linearize the dynamics.
        ts : torch.Tensor, shape=(N - 1,)
            The linearization times.

        Returns
        -------
        Fs : torch.Tensor, shape=(N - 1, n, n)
        Gs : torch.Tensor, shape=(N - 1, n, m)
        Hs : torch.Tensor, shape=(N - 1, n,)
        """
        assert x_bars.shape == (self.N, self.n)
        assert u_bars.shape == (self.N - 1, self.m)
        assert ts.shape == (self.N - 1,)

        # linearized (affine) dynamics at (x_bar, u_bar, t)
        with torch.no_grad():
            dx_bars = self.dyn(x_bars[0:-1, :], u_bars, ts)  # (N - 1, n)
            As, Bs = self.dyn.jacobian(x_bars[0:-1, :], u_bars, ts)
        Cs_1 = dx_bars
        Cs_2 = (As @ x_bars[:-1, :, None]).squeeze(-1)
        Cs_3 = (Bs @ u_bars[..., None]).squeeze(-1)
        Cs = Cs_1 - Cs_2 - Cs_3  # (N - 1, n)

        assert dx_bars.shape == (self.N - 1, self.n)
        assert As.shape == (self.N - 1, self.n, self.n)
        assert Bs.shape == (self.N - 1, self.n, self.m)

        # converting to F, G, H in the A non-invertible case
        # shape=(N - 1, n + m + 1, n + m + 1)
        block = torch.zeros(
            (self.N - 1, self.n + self.m + 1, self.n + self.m + 1),
            dtype=torch.float64,
            device=device,
        )
        block[:, :self.n, :self.n] = As
        block[:, :self.n, self.n:(self.n + self.m)] = Bs
        block[:, :self.n, -1] = Cs

        expm_block = torch.matrix_exp(self.dt * block)
        Fs = expm_block[..., :self.n, :self.n]  # (N - 1, n, n)
        Gs = expm_block[..., :self.n, self.n:(self.n + self.m)]  # (N - 1, n, m)
        Hs = expm_block[..., :self.n, -1]  # (N - 1, n)

        return Fs, Gs, Hs

    def compute_trajectory(
        self,
        x0: torch.Tensor,
        t: float,
        xs_guess: torch.Tensor | None = None,
        us_guess: torch.Tensor | None = None,
        max_outer_iters: int = 100,
        sqp_tol: float = 1e-5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes the optimal trajectory using sequential quadratic programming.

        Parameters
        ----------
        x0 : torch.Tensor, shape=(n,)
            The initial state.
        t : float
            The current time.
        xs_guess : torch.Tensor | None, shape=(N, n), default=None
            Initial guess for the solver.
        us_guess : torch.Tensor | None, shape=(N - 1, m), default=None
            Initial guess for the solver.
        max_outer_iters : int, default=100
            Maximum number of outer iterations of SQP.
        sqp_tol : float, default=1e-5
            The convergence tolerance for the outer loop.

        Returns
        -------
        xs_opt : torch.Tensor, shape=(N, n)
            The optimized state sequence.
        us_opt : torch.Tensor, shape=(N - 1, m)
            The optimized input sequence.
        """
        assert x0.shape == (self.n,)
        if xs_guess is not None:
            assert xs_guess.shape == (self.N, self.n)
        if us_guess is not None:
            assert us_guess.shape == (self.N - 1, self.m)
        assert isinstance(max_outer_iters, int)
        assert isinstance(sqp_tol, float)
        assert max_outer_iters > 0
        assert sqp_tol > 0.0

        # warm starting SQP
        # [TODO] warm start xs by running the dynamics forward with 0 input
        _ts = torch.linspace(
            t, t + self.dt * (self.N - 1), self.N, dtype=torch.float64, device=device
        )  # (N,)
        ts = _ts[:-1]  # (N - 1,)
        if us_guess is None:
            us_prev = torch.zeros(
                (self.N - 1, self.m), dtype=torch.float64, device=device,
            )  # (N - 1, m)
        else:
            us_prev = us_guess
        if xs_guess is None:
            with torch.no_grad():
                ctrl = PiecewiseConstantController(
                    self.dyn, self.dt, us_prev[None, ...]
                )
                _xs_prev, _ = self.dyn.simulate(
                    x0[None, ...], controller=ctrl, ts=_ts
                )
            xs_prev = _xs_prev.squeeze(0)  # (N, n)
        else:
            xs_prev = xs_guess
        self.xs.value = xs_prev.detach().numpy()
        self.us.value = us_prev.detach().numpy()

        # SQP iterations
        for i in range(max_outer_iters):
            # linearizing dynamics about previous solution trajectory
            _Fs, _Gs, _Hs = self._compute_FGHs(xs_prev, us_prev, ts)
            Fs = _Fs.detach().numpy()
            Gs = _Gs.detach().numpy()
            Hs = _Hs.detach().numpy()

            # setting optimization parameters
            self.x0.value = xs_prev[0, :].detach().numpy()
            for k in range(self.N - 1):
                self.Fs[k].value = Fs[k, ...]
                self.Gs[k].value = Gs[k, ...]
                self.Hs[k].value = Hs[k, ...]

            # solving
            self.prob.solve(
                solver=cp.GUROBI,
                warm_start=True,
                # verbose=True,
                presolve_level=2,
            )
            xs_opt = torch.tensor(self.xs.value, dtype=torch.float64, device=device)
            us_opt = torch.tensor(self.us.value, dtype=torch.float64, device=device)

            # checking convergence
            cond1 = torch.all(torch.linalg.norm(xs_opt - xs_prev, axis=-1) <= sqp_tol)
            cond2 = torch.all(torch.linalg.norm(us_opt - us_prev, axis=-1) <= sqp_tol)
            if cond1 and cond2:
                break

            xs_prev = xs_opt
            us_prev = us_opt

        return xs_opt, us_opt

if __name__ == "__main__":
    from core.systems import Quadrotor
    from core.systems import InvertedPendulum

    # # quadrotor
    # mass = 1.0  # mass
    # I = np.array([1.0, 1.0, 1.0])  # principal moments of inertia
    # kf = 1.0  # thrust factor
    # km = 1.0  # drag factor
    # l = 0.1  # rotor arm length
    # Jtp = None
    # quad = Quadrotor(mass, I, kf, km, l, Jtp)

    # # traj opt
    # N = 2
    # dt = 0.1
    # Q = np.eye(12)
    # R = 0.01 * np.eye(4)
    # Qf = np.eye(12)
    # u_min = np.zeros(4)
    # u_max = None
    # traj_opt = TrajectoryOptimizer(quad, N, dt, Q, R, Qf, u_min, u_max)

    # # optimizing traj
    # x0 = np.zeros(12)
    # t = 0.0
    # xs_opt, us_opt = traj_opt.compute_trajectory(x0, t)

    # inverted pend
    mass = 1.0  # mass
    l = 1.0  # length
    pend = InvertedPendulum(mass, l)

    # traj opt
    N = 21
    dt = 0.25
    Q = np.diag([1e3, 1e2])
    R = 1e-3 * np.eye(1)
    Qf = np.diag([1e7, 1e6])
    u_min = np.array([-2.0])
    u_max = np.array([2.0])
    traj_opt = TrajectoryOptimizer(pend, N, dt, Q, R, Qf, u_min, u_max)

    # optimizing traj
    x0 = torch.tensor([-np.pi, 0.0], dtype=torch.float64, device=device)
    t = 0.0
    # xs_opt, us_opt = traj_opt.compute_trajectory(x0, t)

    xs_intermediate = []
    counter = 0

    import matplotlib.pyplot as plt
    
    def _traj_opt_wrapper(x, t, xt_prev, ut_prev):
        global counter
        xs_guess = xt_prev.squeeze(0) if xt_prev is not None else None
        us_guess = ut_prev.squeeze(0) if ut_prev is not None else None
        xs, us = traj_opt.compute_trajectory(
            x.squeeze(0),
            t,
            xs_guess=xs_guess,
            us_guess=us_guess,
        )
        xs_intermediate.append(x.squeeze(0).detach().numpy())

        # intermediate plots
        if counter % 10 == 0:
            fig, ax = pend.plot_states(
                np.linspace(0.0, dt * (len(xs_intermediate) - 1), len(xs_intermediate)),
                np.stack(xs_intermediate, axis=0),
                color="black",
            )
            pend.plot_states(
                np.linspace(t, t + dt * (N - 1), N),
                xs,
                color="red",
                fig=fig,
                ax=ax,
            )
            plt.show()
        counter += 1
        return xs[None, ...], us[None, ...]

    mpc_ctrl = MPCController(pend, _traj_opt_wrapper)

    xs_sol, _ = pend.simulate(
        x0[None, ...],
        controller=mpc_ctrl,
        ts=torch.linspace(0, 5, 201, dtype=torch.float64, device=device),
    )

    import matplotlib.pyplot as plt
    plt.plot(xs_sol[0, :, 0].detach().numpy(), xs_sol[0, :, 1].detach().numpy())
    plt.show()

    breakpoint()

    # # plotting
    # ts = np.linspace(t, t + dt * (N - 1), N)
    # # breakpoint()
    # fig, ax = pend.plot_states(
    #     ts,
    #     np.stack(xs_opt, axis=0),
    #     color="black",
    # )
    # radius = 0.5
    # # add_points(ax=ax, coordinates=xs_opt[0].detach().numpy(),
    # #            radius=radius, color="blue")
    # # add_points(ax=ax, coordinates=xs_opt[-1].detach().numpy(),
    # #            radius=radius*.8, color="green")
    # pend.plot_states(ts, xs_opt, color="red",
    #                    fig=fig,
    #                    ax=ax)
    # import matplotlib.pyplot as plt
    # plt.show()

    # import matplotlib.pyplot as plt
    # breakpoint()
