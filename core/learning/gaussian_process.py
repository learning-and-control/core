import torch as th
from numpy.core._multiarray_umath import log
from torch import nn
from pathlib import Path
from typing import List
from gpytorch.constraints import GreaterThan

#TODO: GP is coupled to RBF kernel through training method. Refactor later.

# more streamlined compatibility with numpy
th.set_default_dtype(th.float64)

@th.jit.script
def pi():
    return 3.141592653589793

def diff_mat(x1, x2):
    return (x1.unsqueeze(1) - x2.unsqueeze(0))[:, :, 0]

class PeriodicKernel(nn.Module):
    def __init__(self, p_prior, active_dims=None, learn_period=False):
        super().__init__()
        # self.p = nn.Parameter(th.tensor(p_prior), requires_grad=False)
        self.p = nn.Parameter(th.tensor(p_prior, dtype=th.float64),
                              requires_grad=learn_period)
        self._length_scale = nn.Parameter(th.tensor(0.0))
        self._signal_variance = nn.Parameter(th.tensor(0.0))

    def forward(self, x1, x2):
        assert x1.shape[1] == 1, "[ERROR] Only 1D inputs supported"
        diff = diff_mat(x1, x2)
        sin_term = th.sin(pi() * diff.div(self.p))
        sin_length = sin_term.div(self.length_scale)
        return th.exp(-2 * sin_length.pow(2)) * \
               self.signal_variance

    def ddx1(self, x1, x2):
        K = self.forward(x1, x2)
        diff = diff_mat(x1, x2)
        sin_factor = th.sin(2*pi() * diff.div(self.p))
        sin_factor_l = -2 * pi() * sin_factor.div(self.length_scale.pow(2) * self.p)
        #WARNING: This is a hack.
        return K * sin_factor_l

    def ddx2(self, x1, x2):
        return - self.ddx1(x1, x2)

    def d2dx1x2(self, x1, x2):
        K = self.forward(x1, x2)
        diff = diff_mat(x1, x2)
        c1 = 2 * self.length_scale.pow(2) * th.cos(2 * pi() * diff.div(self.p))
        c2 = th.cos(4 * pi() * diff.div(self.p))
        constants = (2 * pi() * pi())
        return K * constants * (c1 + c2 - 1).div(self.length_scale.pow(4) * self.p.pow(2))

    @property
    def length_scale(self):
        return self._length_scale.exp()

    @property
    def signal_variance(self):
        return self._signal_variance.exp()

    def __str__(self):
        return f"PeriodicKernel(l={self.length_scale},p={self.p}," \
               f"v={self.signal_variance})"

class MultiplicativeKernel(nn.Module):

    def __init__(self, kernels,  active_dims):
        super().__init__()
        assert len(kernels) == len(active_dims)
        self.kernels = nn.ModuleList(kernels)
        self.active_dims = active_dims

    def forward(self, x1, x2):
        product = th.ones(x1.shape[0], x2.shape[0])
        for i, kernel in enumerate(self.kernels):
            product = product * kernel(
                x1[:, self.active_dims[i]],
                x2[:, self.active_dims[i]])
        return product

    def _augment_jacobian(self, x1, x2, j_small, ad : List[int]):
        if j_small.dim() == 2:
            j_small = j_small.unsqueeze(-1)
        augmented_jacobian = th.zeros(x1.shape[0], x2.shape[0], x1.shape[1])
        augmented_jacobian[:,:, ad] += j_small
        return augmented_jacobian

    def _ddx(self, x1, x2, var_x:int) -> th.Tensor:

        J = th.zeros(x1.shape[0], x2.shape[0], x1.shape[1], dtype=th.float64)
        prod_terms = [kernel(
                x1[:, self.active_dims[i]],
                x2[:, self.active_dims[i]]) for i, kernel in enumerate(self.kernels)]

        for i, kernel in enumerate(self.kernels):
            if var_x == 1:
                dterm_small = kernel.ddx1(x1[:, self.active_dims[i]],
                                                   x2[:, self.active_dims[i]])
            else:# var_x ==2:
                assert var_x == 2
                dterm_small = kernel.ddx2(x1[:, self.active_dims[i]],
                                                   x2[:, self.active_dims[i]])
            dterm = self._augment_jacobian(x1, x2, dterm_small, self.active_dims[i])
            prodterm = th.ones(x1.shape[0], x2.shape[0], dtype=th.float64)
            for j, x in enumerate(prod_terms):
                if j!= i :
                    prodterm = prodterm * prod_terms[j]
            # prod_rule = th.einsum('ij,ijmnpq,pqz -> iqz', prodterm, T, dterm)
            prod_rule = dterm *prodterm.unsqueeze(-1).expand_as(dterm)
            J += prod_rule
        return J

    def ddx1(self, x1, x2) -> th.Tensor:
        return self._ddx(x1, x2,  var_x=1)

    def ddx2(self, x1, x2) -> th.Tensor:
        return self._ddx(x1, x2, var_x=2)

    def d2dx1x2(self, x1, x2):
        n_inputs = x1.shape[1]
        H = th.zeros(x1.shape[0], x2.shape[0], n_inputs, n_inputs, dtype=th.float64)

        d2_terms: List[th.Tensor] = []
        dx1_terms: List[th.Tensor] = []
        dx2_terms: List[th.Tensor] = []
        k_terms: List[th.Tensor] = []
        for i, kernel in enumerate(self.kernels):

            k_terms += [kernel(x1[:, self.active_dims[i]],
                                        x2[:, self.active_dims[i]])]

            dx1_term = self._augment_jacobian(x1, x2,
                             j_small=kernel.ddx1(
                                 x1[:,self.active_dims[i]], x2[:, self.active_dims[i]]),
                            ad=self.active_dims[i])
            dx2_term = self._augment_jacobian(x1, x2,
                             j_small=kernel.ddx2(
                                 x1[:,self.active_dims[i]], x2[:, self.active_dims[i]]),
                            ad=self.active_dims[i])

            d2_term = th.zeros_like(H, dtype=th.float64)
            h_small = kernel.d2dx1x2(x1[:, self.active_dims[i]],
                                    x2[:, self.active_dims[i]])
            if h_small.dim() == 2:
                h_small = h_small.unsqueeze(-1).unsqueeze(-1)
            mesh_1, mesh_2, mesh_3, mesh_4 = th.meshgrid(
                th.arange(0, x1.shape[0]),
                th.arange(0, x2.shape[0]),
                th.tensor(self.active_dims[i]),
                th.tensor(self.active_dims[i]))
            d2_term[mesh_1, mesh_2, mesh_3, mesh_4] = h_small
            d2_terms += [d2_term]
            dx1_terms += [dx1_term]
            dx2_terms += [dx2_term]

        #d/dx1x2 terms
        for i in range(len(self.kernels)):
            prod_other = th.ones(x1.shape[0], x2.shape[0])
            for j in range(len(self.kernels)):
                if i != j:
                    prod_other = prod_other * k_terms[j]
            H += prod_other[:,:,None,None].expand_as(d2_terms[i]) * d2_terms[i]
            # H += th.einsum('ij,ijmnpq,pqzx -> iqzx', prod_other, T, d2_terms[i])
        #d/dx1:T:d/dx2 terms
        for i in range(len(self.kernels)):
            for j in range(len(self.kernels)):
                if i == j: continue
                prod_other = th.ones(x1.shape[0], x2.shape[0])
                for k in range(len(self.kernels)):
                    if i != k and j != k:
                        prod_other = prod_other * k_terms[k]
                # h_added = th.einsum('ij,ijmnpq,pqz -> iqz', prod_other, T, dx1_terms[i])
                h_added = prod_other.unsqueeze(-1).expand_as(dx1_terms[i]) * dx1_terms[i]
                # h_added = th.einsum('ijx,ijmnpq,pqz -> iqxz', h_added, T, dx2_terms[j])
                h_added = h_added.unsqueeze(-1) * dx2_terms[j].unsqueeze(2)
                H += h_added
        return H

class AdditiveKernel(nn.Module):
    def __init__(self, kernels, active_dims):
        super().__init__()
        assert len(kernels) == len(active_dims)
        self.kernels = nn.ModuleList(kernels)
        self.active_dims = active_dims

    def forward(self, x1, x2=None):
        if x2 is None:
            x2 = x1
        sum = th.zeros(x1.shape[0], x2.shape[0])
        for i in range(len(self.kernels)):
            sum = sum + self.kernels[i](
                x1[:, self.active_dims[i]],
                x2[:, self.active_dims[i]])
        return sum

    def ddx1(self, x1, x2):
        if x2 is None:
            x2 = x1
        J = th.zeros(x1.shape[0], x2.shape[0], x1.shape[1])
        for i in range(len(self.kernels)):
            term = self.kernels[i].ddx1(
                x1[:, self.active_dims[i]],
                x2[:, self.active_dims[i]])

            if term.dim() == 1:
                term = term[:, None, None]
            elif term.dim() == 2:
                term = term.unsqueeze(-1)
            J[:, :, self.active_dims[i]] += term
        return J

    def ddx2(self, x1, x2):
        if x2 is None:
            x2 = x1
        J = th.zeros(x1.shape[0], x2.shape[0], x2.shape[1])
        for i in range(len(self.kernels)):
            term = self.kernels[i].ddx2(
                x1[:, self.active_dims[i]],
                x2[:, self.active_dims[i]])
            if term.dim() < J.dim():
                term = term.unsqueeze(-1)
            J[:, :, self.active_dims[i]] += term
        return J

    def d2dx1x2(self, x1, x2):
        if x2 is None:
            x2 = x1
        n_inputs = x1.shape[1]
        H = th.zeros(x1.shape[0], x2.shape[0], n_inputs, n_inputs)

        for i in range(len(self.kernels)):
            term = self.kernels[i].d2dx1x2(x1[:, self.active_dims[i]],
                                         x2[:, self.active_dims[i]])
            if term.dim() == 0:
                H[:, :, self.active_dims[i], self.active_dims[i]] += term
                continue
            mesh = th.meshgrid([
                th.arange(0, H.shape[0]),
                th.arange(0, H.shape[1]),
                th.tensor(self.active_dims[i]),
                th.tensor(self.active_dims[i])])
            if term.dim() == 2:
                H[mesh] += term[:, :, None, None]
            elif term.dim() == 4:
                H[mesh] += term
            else:
                raise("[ERROR] Hessian term with invalid dimensions.")
        return H

class AffineDotProductKernel(nn.Module):
    def __init__(self, s_idx, m_idx, kernels, last_is_unit=True):
        super().__init__()
        self.kernels = nn.ModuleList(kernels)
        self.s_idx = s_idx
        self.m_idx = m_idx
        self.sTox = dict([(i, self.s_idx[i]) for i, idx in enumerate(
            self.s_idx)])
        self.mTox = dict([(i, self.m_idx[i]) for i, idx in enumerate(
            self.m_idx)])
        self.last_is_unit = 1

    def _separate_affine(self, x):
        return x[:, self.s_idx], x[:, self.m_idx]
    def _build_affine_vec(self, m1, m2):
        if self.last_is_unit:
            return th.cat([m1, th.ones(m1.shape[0],1)], dim=1), \
                   th.cat([m2, th.ones(m2.shape[0], 1)], dim=1)
        return m1, m2

    def forward(self, x1, x2):
        s1, m1 = self._separate_affine(x1)
        s2, m2 = self._separate_affine(x2)
        m1, m2 = self._build_affine_vec(m1, m2)
        # mprime = self._build_affine_vec(m1, m2, swap=True)
        Ks = th.stack([k(s1,s2) for k in self.kernels], dim=0)
        m_prod = th.einsum("ij,kj->jik", m1, m2)
        return (Ks * m_prod).sum(dim=0)
    def ddx1(self, x1, x2):
        s1, m1 = self._separate_affine(x1)
        s2, m2 = self._separate_affine(x2)
        m1, m2 = self._build_affine_vec(m1, m2)
        m_prod  = th.einsum("ij,kj->ikj", m1, m2)
        Ks = th.stack([k(s1, s2) for k in self.kernels], dim=-1)
        dkds1 = th.stack([k.ddx1(s1,s2) for k in self.kernels], dim=-2)
        dds1 = (dkds1 * m_prod.unsqueeze(-1)).sum(dim=-2)
        ddm1 = Ks * m2
        idx_perms = th.zeros((len(self.s_idx) + len(self.m_idx),), dtype=th.long)
        for i, idx in enumerate(self.s_idx):
            idx_perms[i] = idx
        for i, idx in enumerate(self.m_idx):
            idx_perms[i+len(self.s_idx)] = idx
        return th.cat([dds1, ddm1], dim=-1)[:,:,idx_perms]

    def ddx2(self, x1, x2):
        s1, m1 = self._separate_affine(x1)
        s2, m2 = self._separate_affine(x2)
        m1, m2 = self._build_affine_vec(m1, m2)
        m_prod = th.einsum("ij,kj->ikj", m1, m2)
        Ks = th.stack([k(s1, s2) for k in self.kernels], dim=-1)
        dkds2 = th.stack([k.ddx2(s1, s2) for k in self.kernels], dim=-2)
        dds2 = (dkds2 * m_prod.unsqueeze(-1)).sum(dim=-2)
        ddm2 = Ks * m1[:, None, :]
        idx_perms = th.zeros((len(self.s_idx) + len(self.m_idx),), dtype=th.long)
        for i, idx in enumerate(self.s_idx):
            idx_perms[i] = idx
        for i, idx in enumerate(self.m_idx):
            idx_perms[i+len(self.s_idx)] = idx
        return th.cat([dds2, ddm2], dim=-1)[:,:,idx_perms]

    def d2dx1x2(self, x1, x2):
        s1, m1 = self._separate_affine(x1)
        s2, m2 = self._separate_affine(x2)
        m1, m2 = self._build_affine_vec(m1, m2)
        m_prod = th.einsum("ij,kj->ikj", m1, m2)
        Ks = th.stack([k(s1, s2) for k in self.kernels], dim=-1)

        d2kds1s2 = th.stack([k.d2dx1x2(s1, s2) for k in self.kernels], dim=-1)
        d2ds1s2 = (d2kds1s2 * m_prod[:, :, None, None,:]).sum(dim=-1)
        d2dm1m2 = th.einsum("ij, klj -> klij", th.eye(len(self.kernels)), Ks)

        dkds2 = th.stack([k.ddx2(s1, s2) for k in self.kernels], dim=-2)
        d2dm1s2 = m2[None, :, :, None] * dkds2

        dkds1 = th.stack([k.ddx1(s1,s2) for k in self.kernels], dim=-1)
        d2ds1m2 = m1[:, None,  None, :] * dkds1

        cov_tensor = th.cat([ th.cat([d2ds1s2, d2ds1m2], dim=-1),
                              th.cat([d2dm1s2, d2dm1m2], dim=-1)], dim=-2)
        idx_perms = th.zeros((len(self.s_idx) + len(self.m_idx),), dtype=th.long)
        for i, idx in enumerate(self.s_idx):
            idx_perms[i] = self.sTox[i]
        for i, idx in enumerate(self.m_idx):
            idx_perms[i + len(self.s_idx)] = self.mTox[i]
        X, Y = th.meshgrid(idx_perms, idx_perms)
        return cov_tensor[:, :, X, Y]

class RBFKernel(nn.Module):
    def __init__(self, d, ard_num_dims=False):
        super().__init__()
        self.d = d
        if ard_num_dims:
            self._length_scale = nn.Parameter(th.ones((self.d,)))
        else:
            self._length_scale = nn.Parameter(th.tensor(1.))
        self._signal_variance = nn.Parameter(th.tensor(1.))

    def forward(self, x1, x2):
        # if x2 is None:
        #     x2 = x1
        diff = (x1.div(self.length_scale()).unsqueeze(1) - x2.div(self.length_scale()).unsqueeze(0))
        r = diff.pow(2).sum(dim=-1)
        # theta = th.diag(self.length_scale.pow(-2))
        # K_raw = th.einsum('ijk,kk, ijk->ij', diff, theta, diff)
        # K_raw = 0.5 * (K_raw + K_raw.transpose(0,1))
        return self.signal_variance() *  (-0.5*r).exp()

    def ddx1(self, x1, x2):
        diff = (x1.unsqueeze(1) - x2.unsqueeze(0)).div(self.length_scale().pow(2))
        K = self.forward(x1,x2)
        return -th.einsum("ij, ijk -> ijk",  K, diff)
        # r = diff.pow(2).sum(dim=-1)

    def ddx2(self, x1, x2):
        return -self.ddx1(x1,x2)

    def d2dx1x2(self, x1, x2):
        K = self.forward(x1, x2)
        theta = th.eye(x1.shape[-1]).div(self.length_scale().pow(2))
        diff = (x1.unsqueeze(1) - x2.unsqueeze(0)).div(self.length_scale().pow(2))
        #outerproduct of each differnce
        outer = th.einsum('ijk,ijl -> ijkl', diff, diff)
        #multiply each K with the difference between theta and outer
        returns = th.einsum("ij,ijkl -> ijkl",K,(theta - outer))
        return returns

    def length_scale(self):
        return self._length_scale.exp()

    def signal_variance(self):
        return self._signal_variance.exp()

    def __str__(self):
        return f"RBFKernel(l={self.length_scale()},v={self.signal_variance()})"

class GaussianProcess(nn.Module):

    def __init__(self, train_x, train_y, kernel,
                 K=None, L=None, alpha=None):
        super().__init__()
        if train_x is None:
            self.n_samples = None
        else:
            self.n_samples = train_x.shape[0]

        if train_y is None:
            self.n_outs = None
        else:
            self.n_outs = train_y.shape[1]
        self.kernel = kernel
        self.register_buffer('train_x', train_x)
        self.register_buffer('train_y', train_y)

        self._noise_variance = nn.Parameter(th.tensor(0.2))
        self.register_buffer('K', K)
        self.register_buffer('L', L)
        self.register_buffer('alpha', alpha)

    def noise_variance(self):
        return self._noise_variance.exp()

    def marginal_log_likelihood(self):
        return -0.5*(self.train_y.transpose(1,0) @ self.alpha).diag().sum() \
                 -th.log(self.L.diag()).sum() \
                 - (self.n_samples/2) * log(2*pi())* self.train_y.shape[1]

    def _fit_data(self):
        # Fit data
        self.K = self.kernel(self.train_x, self.train_x) + \
                 self.noise_variance() * th.eye(self.n_samples,
                                                device=self.train_x.device)
        self.L = th.cholesky(self.K)
        self.alpha = th.cholesky_solve(self.train_y, self.L)

    def train_model(self, n_iters=200, lr=0.5):
        self.train()
        opt = th.optim.Adam([{'params': self.parameters()}], lr=lr)
        for i in range(n_iters):
            opt.zero_grad()
            self._fit_data()
            loss = - self.marginal_log_likelihood()
            print(self.kernel)
            loss.backward()
            print(f'Iter {i} | Loss: {loss.item()}| '
                  f'Noise Var: {self.noise_variance().item()}')
            opt.step()
        opt.zero_grad()
        self._fit_data()
        self.eval()

    def add_samples(self, X, Y):
        self.train_x = th.cat((self.train_x, X), dim=0)
        self.train_y = th.cat((self.train_y, Y), dim=0)
        self.n_samples += X.shape[0]
        self._fit_data()

    def forward(self, x):
        n_inputs = x.shape[0]
        kstar = self.kernel(self.train_x, x)
        mean = kstar.T @ self.alpha
        v = th.cholesky_solve(kstar, self.L)
        cov = self.kernel(x, x) - kstar.T @ v  + self.noise_variance() * th.eye(x.shape[0])
        idx = th.stack(
            th.meshgrid(th.arange(cov.shape[0], dtype=th.long),
                        th.arange(cov.shape[1], dtype=th.long)),
            dim=-1, ).flatten(0,1)
        vals = cov[idx[:,0], idx[:,1]]
        vals = vals.repeat(self.n_outs)
        idx = th.cat([
            th.cat([idx , i*th.ones((idx.shape[0],2), dtype=th.long)], dim=1) \
            for i in range(self.n_outs)], dim=0)
        sparse_cov = th.sparse_coo_tensor(
                             indices=idx.permute(1,0),
                             values=vals,
                             size=(n_inputs, n_inputs, self.n_outs,
                                   self.n_outs),
                            dtype=cov.dtype,
                            device=cov.device)
        return mean, sparse_cov
        # return mean, cov.unsqueeze(2).expand(n_inputs, n_inputs, self.n_outs).diag_embed()

    def posterior(self, x):
        mean, cov = self.__call__(th.from_numpy(x))
        return mean.detach().numpy(), cov.detach().nmumpy()

    @th.jit.export
    def ddx(self, x):
        kstar = self.kernel.ddx1(x, self.train_x)
        kstar_T = self.kernel.ddx2(self.train_x, x)
        mean = th.einsum("jik, ia ->jka", kstar, self.alpha)
        cov = self.kernel.d2dx1x2(x,x) - \
              th.einsum("ikj, akl -> iajl",
                        kstar,
                        th.cholesky_solve(kstar_T.transpose(1,0), self.L))#\
        # \
        #       + th.einsum("ijk, alk -> iajl", mean, mean)
        # covariance fixup Start
        cov_true = cov.permute(0,2,1,3).reshape(cov.shape[0] * cov.shape[2],
                                             cov.shape[1] * cov.shape[3])
        cov_true = (cov_true + cov_true.transpose(1,0)) / 2
        eigs, V = cov_true.symeig(eigenvectors=True)
        fixed_eigs = ((eigs + abs(eigs)) / 2) + 1e-6 *(eigs <= 1e-6)
        fixed_cov = V @ th.diag(fixed_eigs) @ V.transpose(1,0)
        cov = fixed_cov.reshape(cov.shape[0], cov.shape[2],
                                cov.shape[1], cov.shape[3]).permute(0,2,1,3)
        #debug statements
        if eigs.min() <= -1e-5:
            print("Covariance: ")
            print(cov_true)
            print("Eigs: ")
            print(eigs)
            hess = self.kernel.d2dx1x2(x,x).permute(0,2,1,3).reshape(cov.shape[0] * cov.shape[2],
                                             cov.shape[1] * cov.shape[3])
            hess = (hess + hess.transpose(1,0)) /2
            hess_eigs, _ = hess.symeig(eigenvectors=True)
            print("Hessian Eigs:")
            print(hess_eigs)
            raise Exception('[ERROR] Covariance Indefinite to tolerance. ')
        # covariance fixup END
        # cov_true = cov.permute(0, 2, 1, 3).reshape(cov.shape[0] * cov.shape[2],
        #                                            cov.shape[1] * cov.shape[3])
        # cov_true.symeig()[0].min()
        #each output dim has a covariance
        # cov_dense = cov.unsqueeze(-1).expand(cov.shape + (self.n_outs,)).diag_embed()
        idx = th.stack(
            th.meshgrid(th.arange(cov.shape[0], dtype=th.long),
                    th.arange(cov.shape[1], dtype=th.long),
                    th.arange(cov.shape[2], dtype=th.long),
                    th.arange(cov.shape[3], dtype=th.long)),dim=-1).flatten(0, -2)
        vals = cov[idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]]
        vals = vals.repeat(self.n_outs)
        idx = th.cat(
            [th.cat([idx, i * th.ones((idx.shape[0], 2), dtype=th.long)], dim=1) for i in
             range(self.n_outs)], dim=0)
        cov = th.sparse_coo_tensor(
                             indices=idx.permute(1,0),
                             values=vals,
                             size=cov.shape + (self.n_outs, self.n_outs),
                            dtype=cov.dtype,
                            device=cov.device)
        return mean.transpose(2,1), cov

    def ddx_posterior(self, x):
        mean, cov = self.ddx(th.from_numpy(x))
        return mean.detach().numpy(), cov.detach().numpy()

    def load_state_dict(self, state_dict,
                        strict: bool = True):
        super(GaussianProcess, self).load_state_dict(state_dict, strict=False)
        self.train_x = state_dict['train_x']
        self.train_y = state_dict['train_y']
        self.K = state_dict['K']
        self.L = state_dict['L']
        self.alpha = state_dict['alpha']
        self.n_samples = self.train_x.shape[0]
        self.n_outs = self.train_y.shape[1]


def save_gp(gp: GaussianProcess, dir : Path):
    th.save(gp.state_dict(), dir)

def load_gp(dir: Path, kernel):
    gp = GaussianProcess(None, None, kernel)
    gp.state_dict()
    state_dict = th.load(dir)
    gp.load_state_dict(th.load(dir))
    gp.eval()
    return gp

class GPScaler:
    def __init__(self, xmins, xmaxs, wraps=None):
        self.xmins = xmins
        self.xmaxs = xmaxs
        self.deltas = xmaxs - xmins
        self.wraps = wraps

    def _wrapped_dims(self, X):
        w_min = self.xmins[self.wraps]
        w_delta = self.deltas[self.wraps]
        return ((X[:, self.wraps] - w_min) % w_delta) + w_min
    def _wrap(self, X):
        if self.wraps is None or not self.wraps.any().item():
            return X
        after_wrap = th.zeros_like(X)
        after_wrap[:, ~self.wraps] = X[:,~self.wraps]
        after_wrap[:, self.wraps] = self._wrapped_dims(X)
        return after_wrap
    def transform(self, X):
        X_wrapped = self._wrap(X)
        return (X_wrapped - self.xmins).div(self.deltas) * 2 - 1

    def inverse(self, Y):
        return  ((Y +1 ) * self.deltas /2) + self.xmins

class ScaledGaussianProcess(GaussianProcess):
    def __init__(self, train_x, train_y, kernel, x_scaler=None, y_scaler=None):
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        train_x_scaled, train_y_scaled = self._scale_guard(train_x, train_y)
        super().__init__(train_x_scaled, train_y_scaled, kernel)

    def _scale_guard(self, X=None, Y=None):
        X_scaled = X
        Y_scaled = Y
        if self.x_scaler is not None and X is not None:
            X_scaled = self.x_scaler.transform(X)
        if self.y_scaler is not None and Y is not None:
            Y_scaled = self.y_scaler.transform(Y)
        return X_scaled, Y_scaled

    def _descale_guard(self, X_scaled=None, Y_scaled=None):
        X = X_scaled
        Y = Y_scaled
        if self.x_scaler is not None and X_scaled is not None:
            X = self.x_scaler.inverse(X)
        if self.y_scaler is not None and Y_scaled is not None:
            Y = self.y_scaler.inverse(Y)
        return X, Y

    def add_samples(self, X, Y):
        X_scaled, Y_scaled = self._scale_guard(X, Y)
        super(ScaledGaussianProcess, self).add_samples(X_scaled, Y_scaled)

    def forward(self, x):
        x_scaled, _ = self._scale_guard(x)
        y_scaled, cov_scaled = super(ScaledGaussianProcess, self).forward(x_scaled)
        y = self._descale_guard(None, y_scaled)[1]
        #TODO: WARNING this is a hack sparse tensor element-wise product
        # missing from Pytorch for now
        cov_scaled=cov_scaled.to_dense()
        return y, cov_scaled * (self.y_scaler.deltas.pow(2) / 4. ).diag_embed()

    def ddx(self, x):
        x_scaled, _ = self._scale_guard(x)
        ddx_scaled, cov_ddx_scaled = super(ScaledGaussianProcess, self).ddx(x_scaled)
        #TODO: WARNING this is a hack sparse tensor element-wise product
        # missing from Pytorch for now
        cov_ddx_scaled = cov_ddx_scaled.to_dense()
        cov = cov_ddx_scaled.div(self.x_scaler.deltas[None, None, :, None, None])
        cov = cov.div(self.x_scaler.deltas[None, None, None, :, None, None])
        cov = cov * self.y_scaler.deltas.pow(2)
        ddx = ddx_scaled.div(self.x_scaler.deltas) * self.y_scaler.deltas[None,:,None]
        return ddx, cov

