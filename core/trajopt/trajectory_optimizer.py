from enum import Enum
from core.dynamics.affine_dynamics import AffineDynamics
from core.controllers import PiecewiseConstantController
import torch as th
import torch.nn as nn
import torch.linalg as la
from cvxpy import ECOS, Variable, Minimize, sum_squares, Problem, quad_form, Parameter
from cvxpy.atoms import diag, abs
from cvxpylayers.torch import CvxpyLayer
from scipy import optimize
from itertools import count

"""TODOs

- obstacle parameterization + constraints (integer programming?)
- get rid of cvxpylayers
- 
"""

class TrajectoryOptimizer(nn.Module):
    def __init__(self, T, h_k,
                 dynamics: AffineDynamics,
                 u_min = None,
                 u_max = None,
                 max_delta_x = None,
                 max_delta_u = None,
                 trapezoidal=False):
        super().__init__()
        self.T = T
        self.h_k = h_k
        self.dyn = dynamics
        self.m = dynamics.m
        self.n = dynamics.n
        self.state_cost = None
        self.terminal_cost = None
        # variable declarations
        self.xt = Variable((self.T, self.n))


        self.x0 = Parameter((self.n,))
        self.R_sqrt = Parameter((self.m, self.m), PSD=True)
        self.Q_sqrt = Parameter((self.n, self.n), PSD=True)
        self.Qf_sqrt = Parameter((self.n, self.n), PSD=True)
        self.constraints = [self.xt[0] == self.x0]
        self.trapezoidal = trapezoidal
        if not trapezoidal:
            self.ut = Variable((self.T-1, self.m))
            self.At = [Parameter((self.n, self.n)) for _ in range(self.T-1)]
            self.Bt = [Parameter((self.n, self.m)) for _ in range(self.T-1)]
            self.Ct = Parameter((self.T-1, self.n))
            for t in range(0, self.T-1):
                # dynamic constraints
                self.constraints += [
                    self.xt[t+1] == self.At[t] @ self.xt[t] + self.Bt[t] @ self.ut[t] + self.Ct[t]
                ]
        else:
            self.ut = Variable((self.T, self.m))
            self.At = [Parameter((self.n, self.n)) for _ in range(self.T)]
            self.Bt = [Parameter((self.n, self.m)) for _ in range(self.T)]
            self.Ct = Parameter((self.T, self.n))
            for t in range(0, self.T-1):
                # dynamic constraints
                drift = self.At[t+1] @ self.xt[t+1] + self.At[t] @ self.xt[t]
                ctrl = self.Bt[t+1] @ self.ut[t+1] + self.Bt[t] @ self.ut[t]
                offset = self.Ct[t+1] + self.Ct[t]
                self.constraints += [
                    self.xt[t+1] - self.xt[t] == 0.5*h_k*(drift + ctrl + offset )
                ]
        self.costs = [
            # sum_squares(self.R_sqrt @ self.ut.T),
            sum_squares(self.Q_sqrt @ self.xt.T),
            sum_squares(self.Qf_sqrt @ self.xt[-1].T)]
        n_controls = self.ut.shape[0]
        if u_min is not None:
            if u_min.ndim == 1:
                u_min = u_min.unsqueeze(0).repeat(n_controls, 1)
            u_min = u_min.detach().numpy()
            self.constraints += [self.ut >= u_min]
        if u_max is not None:
            if u_max.ndim == 1:
                u_max = u_max.unsqueeze(0).repeat(n_controls, 1)
            u_max = u_max.detach().numpy()
            self.constraints += [self.ut <= u_max]
        self.max_delta_x = max_delta_x
        self.max_delta_u = max_delta_u
        if max_delta_x is not None:
            self.xt_prev = Parameter((self.T, self.n))
            self.constraints += [abs(self.xt - self.xt_prev) <= max_delta_x]
        if max_delta_u is not None:
            self.ut_prev = Parameter(self.ut.shape)
            self.constraints += [abs(self.ut - self.ut_prev) <= max_delta_u]
        problem = Problem(Minimize(sum(self.costs)), self.constraints)
        assert problem.is_dpp()
        self.cvx_layer = CvxpyLayer(problem=problem,
                                    parameters=self.get_cvx_parameters(),
                                    variables=self.get_cvx_variables())
        self.problem = problem
    def get_cvx_parameters(self):
        params = [self.x0,
                  # self.R_sqrt,
                  self.Q_sqrt, self.Qf_sqrt]
        params += self.At
        params += self.Bt
        params += [self.Ct]
        if self.max_delta_x is not None:
            params += [self.xt_prev]
        if self.max_delta_u is not None:
            params += [self.ut_prev]
        return params
    def get_cvx_variables(self):
        return [self.xt, self.ut]

    def make_continuous_linear_system_approx_constraints(self, b_yt_prev,
                                                         b_ut_prev, b_xf):
        #time-varying dynamics not currently supported
        #last time step is not included in the dynamics constraints
        b_xt_prev = b_yt_prev + b_xf
        prev_shape = (b_xt_prev.shape[0], self.T-1)
        bt_x_prev = b_xt_prev[:, :-1, :].flatten(0, 1)
        bt_u_prev = b_ut_prev[:, :, :].flatten(0, 1)
        (bAt, bBt) = self.dyn.jacobian(bt_x_prev, bt_u_prev, 0)
        bexpAt = th.matrix_exp(bAt * self.h_k)
        bC = self.dyn(bt_x_prev, bt_u_prev, 0)[..., None] - bAt @ bt_x_prev[...,None] -bBt @ bt_u_prev[...,None]
        bAinv = th.pinverse(bAt)
        CtnToDiscreteMap = bAinv @ (bexpAt - th.eye(*bAt.shape[-2:])[None])
        bDiscB = CtnToDiscreteMap @ bBt
        bDiscC = CtnToDiscreteMap @ bC

        bxf = b_xf[..., None]
        bDiscC = bDiscC + (bexpAt @ bxf) - bxf

        bDiscA = bexpAt.unflatten(0, prev_shape)
        bDiscB = bDiscB.unflatten(0, prev_shape)
        bDiscC = bDiscC.unflatten(0, prev_shape)
        return bDiscA, bDiscB, bDiscC

    def make_trapezoidal_dynamics_constraints(self, b_yt_prev, b_ut_prev, b_xf):
        b_xt_prev = b_yt_prev + b_xf
        prev_shape = (b_xt_prev.shape[0], self.T)
        bt_x_prev = b_xt_prev.flatten(0, 1)
        bt_u_prev = b_ut_prev.flatten(0, 1)
        (bAt, bBt) = self.dyn.jacobian(bt_x_prev, bt_u_prev, 0)
        bft = self.dyn(bt_x_prev, bt_u_prev, 0)

        bCt = bft[..., None] - bAt @ bt_x_prev[...,None] - bBt @ bt_u_prev[...,None]

        bAt = bAt.unflatten(0, prev_shape)
        bBt = bBt.unflatten(0, prev_shape)
        bCt = bCt.unflatten(0, prev_shape)

        return bAt, bBt, bCt


    def _warmstart(self, x0, ts, ws_xt=None, ws_ut=None, force_integrate=False):
        n_batch = x0.shape[0]
        # Simulate zeroinput warmstart if not none is provided
        if ws_ut is not None:
            ut_prev = ws_ut
        else:
            ut_prev = th.zeros((n_batch, self.ut.shape[0], self.m))
        if ws_xt is not None and not force_integrate:
            xt_prev = ws_xt
        else:
            ctrl = PiecewiseConstantController(self.dyn, self.h_k, ut_prev)
            xt_prev, _ = self.dyn.simulate(x0, controller=ctrl, ts=ts)
        return xt_prev, ut_prev

    def eval(self, Q_sqrt, R_sqrt, Qf_sqrt, x0, xf,
             ws_xt=None, ws_ut=None,
             max_cvx_iters=100, converge_tol=1e-5, solver_opts=None):
        ts = th.arange(self.T) * self.h_k
        if Q_sqrt.ndim == 2:
            Q_sqrt = Q_sqrt.unsqueeze(0)
        if R_sqrt.ndim == 2:
            R_sqrt = R_sqrt.unsqueeze(0)
        if Qf_sqrt.ndim == 2:
            Qf_sqrt = Qf_sqrt.unsqueeze(0)
        if xf.ndim == 1:
            xf = xf.unsqueeze(0)
        xt_prev, ut_prev = ws_xt, ws_ut
        xt_prev, ut_prev = self._warmstart(x0, ts, xt_prev, ut_prev,
                                           force_integrate=True)
        yt_prev = xt_prev - xf
        for i in count():
            if i >= max_cvx_iters:
                break
            xt_prev, ut_prev = self._warmstart(x0, ts, yt_prev + xf, ut_prev,
                                               force_integrate=True)
            first_step_error=(yt_prev + xf - xt_prev)[:, 0].norm(dim=-1).max().item()

            yt_prev = xt_prev - xf

            if not self.trapezoidal:
                bAs, bBs, bCs = \
                    self.make_continuous_linear_system_approx_constraints(
                        yt_prev, ut_prev, xf)
            else:
                bAs, bBs, bCs = self.make_trapezoidal_dynamics_constraints(
                    yt_prev, ut_prev, xf)
            n_batch = x0.shape[0]
            params = (x0 - xf,
                      # R_sqrt.repeat(n_batch, 1, 1),
                      Q_sqrt.repeat(n_batch, 1, 1),
                      Qf_sqrt.repeat(n_batch, 1, 1),
                      ) + bAs.unbind(dim=1) + bBs.unbind(dim=1) + (bCs[...,0],)
            if self.max_delta_x is not None:
                params += (yt_prev,)
            if self.max_delta_u is not None:
                params += (ut_prev,)

            param_vars = [self.x0, self.Q_sqrt, self.Qf_sqrt] + self.At + self.Bt + [self.Ct]
            for p_var, p_tensor in zip(param_vars, params):
                p_var.value = p_tensor[0].detach().cpu().numpy()
            self.problem.solve(solver='OSQP', verbose=True)

            tmp =  self.x0.value.copy()
            yt[0,0].detach().cpu().numpy() -  self.xt[0].value


            yt, ut = self.cvx_layer(*params, solver_args={
                # 'solve_method': 'ECOS',
                # # 'feastol': 1e-3,
                # 'verbose': True,
                # 'max_iters': 200000,
                'solve_method': 'SCS',
                'verbose': True,
                'max_iters': 50000,
                'acceleration_lookback': 10,
                'use_indirect': True,
                'normalize': True,
                # 'use_quad_obj': False,
                'eps': 1e-6
            })

            # line_search_p = th.linspace(0, 1, 100)[:, None, None, None]
            # ls_shape = (line_search_p.shape[0], n_batch)
            # ut_ls = line_search_p * ut[None] + (1 - line_search_p) * ut_prev[None]
            # x0_ls = x0[None].repeat(line_search_p.shape[0], 1, 1)
            # ctrl = PiecewiseConstantController(self.dyn, self.h_k, ut_ls.flatten(0, 1))
            # xt_ls, _ = self.dyn.simulate(x0_ls.flatten(0, 1), controller=ctrl, ts=ts)
            # err_ls = (Qf_sqrt @ xt_ls[..., -1, :, None])[..., 0].square().sum(dim=-1)
            # # err_ls = err_ls + (Q_sqrt @ xt_ls[..., :-1,:, None])[..., 0].square().sum(dim=(-1,-2))
            # ls_idx = err_ls.unflatten(0, ls_shape).argmin(dim=0)
            # xt_ls = xt_ls.unflatten(0, ls_shape)
            # xt = xt_ls[ls_idx].diagonal(dim1=0, dim2=1).permute(2,0,1)
            # ut = ut_ls[ls_idx].diagonal(dim1=0, dim2=1).permute(2,0,1)

            x_update_size = (yt_prev - yt).abs().mean().item()
            u_update_size = (ut_prev - ut).abs().mean().item()
            yt_prev = yt.clone()
            ut_prev = ut.clone()
            print(f'[INFO] Iteration {i}: update size: (x: {x_update_size}, '
                  f'u: {u_update_size}, fe: {first_step_error})')
            # check for convergence
            if min(x_update_size, u_update_size) < converge_tol:
                print('[INFO] Convergence')
                break

        if i == max_cvx_iters:
            print("[WARNING] Failed to Converge")
        return yt + xf, ut
