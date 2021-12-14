from core.trajopt import TrajectoryOptimizer
from core.systems import AffineGPSystem
from scipy.linalg import expm
from numpy import array, zeros, zeros_like, eye, ones
from numpy.linalg import pinv, inv, cholesky
from cvxpy import quad_form, reshape, vec, norm, square

class GPTrajectoryOptimizer(TrajectoryOptimizer):
    def __init__(self, T, h_k,
                 dynamics: AffineGPSystem,
                 max_delta_x = None,
                 max_delta_u = None,
                 solver='OSQP',
                 cov_penalty=None):
        super().__init__(T, h_k, dynamics,
                         TrajectoryOptimizer.COLLOCATION_MODE.CTN_ONE_PT,
                         max_delta_x, max_delta_u, solver)
        self.cov_penalty=cov_penalty
        assert dynamics.delta_t == h_k, "[ERROR] Dynamics and Trajopt must be time-slice aligned"

    def make_continuous_linear_system_approx_constraints(self, xt, ut):
        dynamics_constraints = list()
        expAs, expBs, expCs, covs = self.dyn.jacobian_exp(xt.value, ut.value)
        for t in range(self.T - 1):
            dynamics_constraints += [
                xt[t+1] == expAs[t] @ xt[t] + expBs[t] @ ut[t] + expCs[t]
            ]

        if self.cov_penalty is not None:
            for t in range(1, self.T):
                self.var_costs += [quad_form(xt[t] - xt[t].value,
                                             covs[t-1, t-1]*self.cov_penalty)]
        return dynamics_constraints