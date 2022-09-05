import torch as th
from numpy import eye, exp
from numpy.linalg import pinv
from scipy.linalg import logm

from core.dynamics import SystemDynamics, Dynamics

class AffineGPSystem(SystemDynamics, Dynamics):

    def __init__(self, gaussianProcess, n, m, delta_t,
                 dyn_dims=None,
                 ddim_to_dim=None,
                 ddim_to_gp_idx=None,
                 force_delta_mode=False):
        SystemDynamics.__init__(self, n, m)
        self.gp = gaussianProcess
        self.delta_t = delta_t
        if dyn_dims is None:
            self.dyn_dims = [d for d in range(n)]
        else:
            self.dyn_dims = dyn_dims
        self.force_delta_mode = force_delta_mode
        self.delta_mode_on = force_delta_mode or\
                             len(self.dyn_dims) != self.n or\
                             ddim_to_dim is not None
        if ddim_to_dim is None:
            self.ddim_to_dim = {}
            self.ddim_to_gp_idx={}
            if ddim_to_gp_idx is not None:
                raise Exception("[ERROR] ddim_to_gp only valid with "
                                "ddim_to_gp_idx")
        else:
            self.ddim_to_dim = ddim_to_dim
            self.ddim_to_gp_idx = ddim_to_gp_idx
            assert set(self.ddim_to_dim.keys()) == set(self.ddim_to_gp_idx.keys())

    def forward(self, x, u, t):
        A, B, C = self._make_AB(x, u)
        return A @ x +  B @ u + C

    def jacobian_exp(self, xts, uts):
        As, Bs, Cs, cov, _ = self._next_step_info(xts, uts)
        Cs = Cs.squeeze()
        return As, Bs, Cs, cov

    def addFrameDims(self, Js):
        Acols = list()
        for i in range(self.n):
            if i in self.dyn_dims:
                idx = self.dyn_dims.index(i)
                Acols.append(Js[:, :, idx])
            else:
                Acols.append(th.zeros(Js.shape[0], self.n))
        As = th.stack(Acols, dim=2)
        return As

    def step(self, x_0, u_0, t_0, t_f, atol=1e-6, rtol=1e-6):
        delta_t = t_f - t_0
        if abs(self.delta_t - delta_t) < atol:
            xtp1 = self.vec_step(x_0[None, :], u_0[None, :])
            return xtp1[0]
        else:
            return super().step(x_0, u_0, t_0, t_f, atol, rtol)

    def vec_step(self, xts, uts):
        delta_xtp1, _ = self.gp(th.cat([xts[:, self.dyn_dims], uts], dim=1))

        if self.delta_mode_on:
            if not self.ddim_to_dim:
                xtp1 = xts + delta_xtp1
            else:
                xtp1 = xts.clone()
                for k, v in self.ddim_to_dim.items():
                    xtp1[:, [v, k]] += th.stack([
                        self.delta_t * xts[:, k],
                        delta_xtp1[:, self.ddim_to_gp_idx[k]]
                    ], dim=1)
        else:
            xtp1 = delta_xtp1

        return xtp1

    def jacobian(self, x, u, t):
        A, B, _ = self._make_AB(x, u)
        return A, B

    def _make_AB(self, x, u):
        assert x.dim() == 1 and u.dim() == 1, "[ERROR] Vectorization not Supported"

        x = x.unsqueeze(0)
        u = u.unsqueeze(0)
        eAdt, u_factor, nl_residual, _, xtp1 = self._next_step_info(x, u)

        eAdt = eAdt[0]
        u_factor = u_factor[0].detach().numpy()
        nl_residual = nl_residual.squeeze()

        eAdt_np = eAdt.detach().numpy()  # batch size should be 1 here
        A = logm(eAdt_np).real * (1 / self.delta_t)
        u_c_factor = A @ pinv(eAdt_np - eye(self.n))
        B = u_c_factor @ u_factor
        A = th.from_numpy(A)
        B = th.from_numpy(B)
        C = th.from_numpy(u_c_factor) @ nl_residual

        return A, B, C

    def _embed_kin_dim(self, J_dyn, delta_xtp1, eAdt, x):
        assert self.delta_mode_on
        n_samples = J_dyn.shape[0]
        if eAdt.ndim < 2:
            eAdt = eAdt.unsqueeze(0)
        full_eAdt = th.zeros(n_samples, self.n, self.n)
        u_factor = th.zeros(n_samples, self.n, self.m)
        xtp1 = x.clone()
        for k, v in self.ddim_to_dim.items():
            full_eAdt[:, k] = eAdt[:, self.ddim_to_gp_idx[k]]
            full_eAdt[:, v, k] = self.delta_t

            u_factor[:, k] = J_dyn[:, self.ddim_to_gp_idx[k], -self.m:]
            u_factor[:, v] = 0.0
            xtp1[:, [v, k]] += th.stack([
                    self.delta_t * x[:, k],
                    delta_xtp1[:, self.ddim_to_gp_idx[k]]
                ], dim=1)
        return full_eAdt, u_factor, xtp1

    def _next_step_info(self, xts, uts):
        delta_xtp1s, cov = self.gp(th.cat([xts[:, self.dyn_dims], uts], dim=1))
        Js, _ = self.gp.ddx(th.cat([xts[:, self.dyn_dims], uts], dim=1))
        if self.delta_mode_on:
            As = self.addFrameDims(Js)
            if not self.ddim_to_dim:
                xtp1s = xts + delta_xtp1s
                Bs = Js[:, :, -self.m:]
            else:
                As, Bs, xtp1s = self._embed_kin_dim(Js, delta_xtp1s, As, xts)
            As = th.eye(self.n) + As
        else:
            As = Js[:, :, :-self.m]
            xtp1s = delta_xtp1s
            Bs = Js[:, :, -self.m:]
        Cs = xtp1s.unsqueeze(2) - As @ xts.unsqueeze(2) - Bs @ uts.unsqueeze(2)
        return As, Bs, Cs, cov, delta_xtp1s