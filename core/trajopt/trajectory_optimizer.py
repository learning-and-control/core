from enum import Enum
from core.dynamics.affine_dynamics import AffineDynamics
from core.controllers import PiecewiseConstantController
from scipy.linalg import expm
from numpy import array, zeros, zeros_like, eye, ones
from numpy.linalg import norm, pinv, inv
from cvxpy import ECOS, Variable, Minimize, sum_squares, Problem, quad_form
from cvxpy.atoms import diag, abs
from scipy import optimize
from itertools import count

class TrajectoryOptimizer:
    class COLLOCATION_MODE(Enum):
        TRAPEZOIDAL = 1
        HERMITE_SIMPSON = 2
        CTN_ONE_PT = 3

    def __init__(self, T, h_k,
                 dynamics: AffineDynamics,
                 collocation_mode: COLLOCATION_MODE = COLLOCATION_MODE.CTN_ONE_PT,
                 max_delta_x = None,
                 max_delta_u = None,
                 solver='OSQP'):
        self.T = T
        self.h_k = h_k
        self.dyn = dynamics
        self.m = dynamics.m
        self.n = dynamics.n
        self.state_cost = None
        self.terminal_cost = None
        self.collocation_mode = collocation_mode
        self.const_constraints = list()
        self.const_costs = list()
        self.var_costs = list()
        self.xt = None
        self.ut = None
        self.max_delta_x = max_delta_x
        self.max_delta_u = max_delta_u
        self.solver = solver
        # variable declarations
        self.xt = Variable((self.T, self.n))
        self.ut = Variable((self.T, self.m))

    def make_hermite_simpson_dynamics_constraints(self, xt, ut):
        raise NotImplementedError('[ERROR] Hermite-simpson Approx TODO.')

    def make_trapezoidal_dynamics_constraints(self, xt, ut):
        dynamics_constraints = list()
        for t in range(self.T - 1):
            tau = t * self.h_k
            taup1 = (t + 1) * self.h_k
            (At, Bt) = self.dyn.jacobian(xt[t].value, ut[t].value, tau)
            (Atp1, Btp1) = self.dyn.jacobian(xt[t + 1].value, ut[t + 1].value,
                                             taup1)
            ft = At @ (xt[t]- xt[t].value) + Bt @ (ut[t] - ut[t].value) + \
                self.dyn(xt[t].value, ut[t].value, tau)
            ftp1 = Atp1 @ (xt[t + 1] - xt[t+1].value) + Btp1 @ (ut[t + 1] - ut[t+1].value) \
                   + self.dyn(xt[t+1].value, ut[t+1].value, tau)
            dynamics_constraints += [
                xt[t + 1] - xt[t] == 0.5 * self.h_k * (ft + ftp1)]
        return dynamics_constraints

    def make_continuous_linear_system_approx_constraints(self, xt, ut):
        dynamics_constraints = list()
        for t in range(self.T -1):
            tau = t * self.h_k
            (At, Bt) = self.dyn.jacobian(xt[t].value, ut[t].value, tau)
            expAt = expm(At * self.h_k)
            C = self.dyn(xt[t].value, ut[t].value, tau) \
                - At @xt[t].value - Bt @ ut[t].value
            Ainv = pinv(At)
            dynamics_constraints += [
                    xt[t+1] ==\
                    expAt @ xt[t] +\
                    Ainv @  (expAt - eye(*At.shape))@ (Bt @ ut[t] + C)]
        return dynamics_constraints

    def add_input_constraints(self, u_min, u_max):
        for t in range(self.T):
            if u_max is not None:
                self.const_constraints += [self.ut[t] <= u_max]
            if u_min is not None:
                self.const_constraints += [self.ut[t] >= u_min]

    def add_static_quad_cost(self, Q=None, R=None, offset=None):
        if offset is None:
            offset = zeros_like(self.xt[0].value)
        # constant costs
        if R is not None:
            for t in range(self.T):
                self.const_costs += [quad_form(self.ut[t], R)]
        else:
            self.const_costs += [sum_squares(self.ut)]
        if Q is not None:
            for t in range(self.T):
                self.const_costs += [quad_form(self.xt[t] - offset, Q)]

    def add_terminal_cost(self, Q_f, offset=None, R_f=None):
        if offset is None:
            offset = zeros_like(self.xt[-1].value)
        self.const_costs += [quad_form(self.xt[-1]-offset, Q_f)]

    def add_hard_terminal_constraint(self, x_f):
        self.const_constraints += [self.xt[-1] == x_f]

    def add_trust_region_constraint(self):
        trust_region = list()
        if self.max_delta_x is not None:
            for t in range(self.T):
                trust_region += [abs(self.xt[t] - self.xt[t].value) <= self.max_delta_x ]
        if self.max_delta_u is not None:
            for t in range(self.T):
                trust_region += [abs(self.ut[t] - self.ut[t].value) <= self.max_delta_u ]
        return trust_region

    def add_state_box_constraints(self, x_min, x_max):
        box_constraints = list()
        for t in range(self.T):
            box_constraints += [self.xt[t] <= x_max]
            box_constraints += [self.xt[t]  >= x_min]
        self.const_constraints += box_constraints


    def _warmstart(self, x_0, ws_xt=None, ws_ut=None):
        # Simulate zeroinput warmstart if not none is provided
        if ws_ut is not None:
            self.ut.value = ws_ut
        else:
            self.ut.value = zeros((self.T, self.m))
        if ws_xt is not None:
            self.xt.value = ws_xt
        else:
            ctrl = PiecewiseConstantController(self.dyn, self.h_k, self.ut.value)
            ts = array(range(self.T)) * self.h_k
            self.xt.value, _ = self.dyn.simulate(x_0, controller=ctrl, ts=ts)

    def eval(self, x_0, ws_xt=None, ws_ut=None,
             max_cvx_iters=100, converge_tol=1e-5, solver_opts=None):

        ts = array([i for i in range(self.T)])*self.h_k
        self._warmstart(x_0, ws_xt, ws_ut)
        for i in count():
            if i >= max_cvx_iters:
                break
            self.var_costs = []
            xtprev = self.xt.value.copy()
            utprev = self.ut.value.copy()
            constraints = self.const_constraints.copy()
            constraints += [self.xt[0] == x_0]
            if self.collocation_mode == TrajectoryOptimizer.COLLOCATION_MODE.TRAPEZOIDAL:
                constraints += self.make_trapezoidal_dynamics_constraints(self.xt,self.ut)
            elif self.collocation_mode == TrajectoryOptimizer.COLLOCATION_MODE.HERMITE_SIMPSON:
                constraints += self.make_hermite_simpson_dynamics_constraints(self.xt,  self.ut)
            elif self.collocation_mode == TrajectoryOptimizer.COLLOCATION_MODE.CTN_ONE_PT:
                constraints += self.make_continuous_linear_system_approx_constraints(self.xt, self.ut)
            else:
                raise Exception("[ERROR] Invalid Collocation mode.")
            constraints += self.add_trust_region_constraint()
            problem = Problem(Minimize(sum(self.const_costs + self.var_costs)),
                              constraints)
            opts = {} if solver_opts is None else solver_opts
            if solver_opts is None and self.solver == "GUROBI":
                opts = {'Presolve':0, 'NumericFocus':3}
            soln = problem.solve(solver=self.solver, verbose=True, **opts)
            # Be Warned: Gurobi's Presolve has returned broken solutions before
            # soln = problem.solve(verbose=True)
            if problem.status == 'infeasible' or \
                problem.status == "infeasible_inaccurate":
                print("[WARNING] Problem is infeasible."
                      "Solving and trying again")
                ctrl = PiecewiseConstantController(self.dyn, self.h_k,
                                                   utprev)
                xs, us = self.dyn.simulate(x_0, controller=ctrl, ts=ts)
                self.xt.value = xs
                self.ut.value = utprev
                continue
            # elif :
            #     print("[WARNING] Problem presumed infeasible to relaxed "
            #           "numerical tolerance.")

            update_size = \
                ((xtprev - self.xt.value) ** 2).sum() + (
                    (utprev - self.ut.value) ** 2).sum()
            print(f'[INFO] Iteration {i}: update size: {update_size}')
            # check for convergence
            if update_size < converge_tol:
                print('[INFO] Convergence')
                break

        if i == max_cvx_iters:
            print("[WARNING] Failed to Converge")
        return self.xt.value, self.ut.value
