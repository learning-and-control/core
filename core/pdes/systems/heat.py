from torch import arange, cat, complex, cumsum, exp, ones, sparse_coo_tensor, stack, tensor, zeros
from torch.linalg import solve
from torchdiffeq import odeint

from ...dynamics import SystemDynamics


def assemble_block_sparse(rows):
    block_row_heights = [len(row[0]) for row in rows]
    block_col_widths = [elem.shape[1] for elem in rows[0]]

    block_row_offsets = cumsum(tensor([0] + block_row_heights[:-1]), dim=0)
    block_col_offsets = cumsum(tensor([0] + block_col_widths[:-1]), dim=0)

    row_indices = []
    col_indices = []
    values = []
    for row, block_row_offset in zip(rows, block_row_offsets):
        for elem, block_col_offset in zip(row, block_col_offsets):
            elem = elem.coalesce()
            row_indices = row_indices + [elem.indices()[0] + block_row_offset]
            col_indices = col_indices + [elem.indices()[1] + block_col_offset]
            values = values + [elem.values()]

    row_indices = cat(row_indices)
    col_indices = cat(col_indices)
    indices = stack([row_indices, col_indices])
    values = cat(values)
    return sparse_coo_tensor(indices, values, (sum(block_row_heights), sum(block_col_widths)))


class Dirichlet(SystemDynamics):
    def eval_dot(self, xs, us, ts):
        raise NotImplementedError

    def step(self, xs, us, t_0, t_f, atol=1e-6, rtol=1e-6):
        return odeint(lambda t, y: self.eval_dot(y, us, t), xs, tensor([t_0, t_f]), atol=atol, rtol=rtol)[-1]

    def poisson_optimize(self, us):
        forcing_matrix = self.mass_matrix @ self.forcings.T
        transposed_forcing_matrix = forcing_matrix.T.to_sparse()
        forcing_matrix = forcing_matrix.to_sparse()

        Z_u_mu = sparse_coo_tensor(size=self.stiffness_matrix.shape)
        Z_mu_a = sparse_coo_tensor(size=forcing_matrix.shape)
        Z_a_u = sparse_coo_tensor(size=transposed_forcing_matrix.shape)
        Z_a_a = sparse_coo_tensor(size=(len(self.forcings),) * 2)
        A = assemble_block_sparse([
            [self.diffusion * self.stiffness_matrix, Z_u_mu, -forcing_matrix],
            [-2 * self.mass_matrix, self.diffusion * self.stiffness_matrix, Z_mu_a],
            [Z_a_u, transposed_forcing_matrix, Z_a_a]
        ])

        Z_us = zeros(len(self.stiffness_matrix), len(us))
        Z_as = zeros(len(transposed_forcing_matrix), len(us))
        bs = cat([Z_us, -2 * (self.mass_matrix @ us.T), Z_as])

        solution = solve(A.to_dense(), bs)  # TODO: Replace this with sparse solver
        return solution


# Homogeneous (default) and nonhomogeneous Dirichlet problems with finite differences
class FiniteDifference(Dirichlet):
    def __init__(self, interval, diffusion, forcings, bcs=None):
        SystemDynamics.__init__(self, interval.N - 1, len(forcings))
        self.interval = interval
        self.diffusion = diffusion
        self.forcings = forcings[:, 1:interval.N]
        self.bcs = bcs if bcs is not None else zeros(2)
        self.boundary_effect = interval.dirichlet_laplacian_bcs @ self.bcs

        dx = interval.L / interval.N

        row_indices = cat([arange(interval.N - 2), arange(interval.N - 1), arange(1, interval.N - 1)])
        col_indices = cat([arange(1, interval.N - 1), arange(interval.N - 1), arange(interval.N - 2)])
        values = cat([ones(interval.N - 2), 4 * ones(interval.N - 1), ones(interval.N - 2)]) * dx / 6
        self.mass_matrix = sparse_coo_tensor(stack([row_indices, col_indices]), values, (interval.N - 1, interval.N - 1))

        self.stiffness_matrix = -interval.dirichlet_laplacian * dx

    def eval_dot(self, xs, us, ts):
        laplacian = (self.interval.dirichlet_laplacian @ xs.T).T + self.boundary_effect
        forcings = (self.forcings.T @ us.T).T
        return self.diffusion * laplacian + forcings

    def poisson_optimize(self, us):
        return super().poisson_optimize(us[:, 1:-1])


# Periodic problems with finite differences
class FiniteDifferencePeriodic(Dirichlet):
    def __init__(self, interval, diffusion, forcings):
        SystemDynamics.__init__(self, interval.N, len(forcings))
        self.interval = interval
        self.diffusion = diffusion
        self.forcings = forcings[:, :interval.N]

        dx = interval.L / interval.N

        row_indices = cat([arange(interval.N - 1), arange(interval.N), arange(1, interval.N), tensor([0, interval.N - 1])])
        col_indices = cat([arange(1, interval.N), arange(interval.N), arange(interval.N - 1), tensor([interval.N - 1, 0])])
        values = cat([ones(interval.N - 1), 4 * ones(interval.N), ones(interval.N - 1), ones(2)]) * dx / 6
        self.mass_matrix = sparse_coo_tensor(stack([row_indices, col_indices]), values, (interval.N, interval.N))

        periodic_laplacian = interval.periodic_laplacian.to_dense()
        self.stiffness_matrix = -(periodic_laplacian * dx + (1 / (interval.N ** 2))).to_sparse()  # Completely dense

    def eval_dot(self, xs, us, ts):
        laplacian = (self.interval.periodic_laplacian @ xs.T).T
        forcings = (self.forcings.T @ us.T).T
        return self.diffusion * laplacian + forcings

    def poisson_optimize(self, us):
        return super().poisson_optimize(us[:, :-1])


class Spectral(SystemDynamics):
    def poisson_optimize(self, u_hats):
        repeated_laplacian_inverse = self.laplacian_inverse.unsqueeze(-1).repeat(1, len(self.forcing_hats))
        C = repeated_laplacian_inverse * self.forcing_hats.T / self.diffusion
        A = -C.T @ C

        repeated_laplacian_inverse = self.laplacian_inverse.unsqueeze(-1).repeat(1, len(u_hats))
        b = self.forcing_hats @ (repeated_laplacian_inverse * u_hats.T) / self.diffusion

        actions = solve(A, b)
        solution_hats = -C @ actions
        lagrange_multiplier_hats = -2 * repeated_laplacian_inverse * (solution_hats - u_hats.T) / self.diffusion
        return cat([solution_hats, lagrange_multiplier_hats, actions])

# Homogeneous Dirichlet problems with Sine transform
class SineSpectral(Spectral):
    def __init__(self, interval, diffusion, forcings, num_modes=None):
        SystemDynamics.__init__(self, interval.N, len(forcings))
        self.interval = interval
        self.diffusion = diffusion
        self.num_modes = num_modes if num_modes is not None else interval.N - 1

        self.laplacian = interval.sine_truncate(interval.sine_spec_laplacian, self.num_modes)
        self.laplacian_inverse = 1 / self.laplacian

        forcing_hats = interval.sine_transform(forcings)
        self.forcing_hats = interval.sine_truncate(forcing_hats, self.num_modes)
        self.forcings = interval.inverse_sine_transform(interval.sine_extend(self.forcing_hats))

    def step(self, xs, us, t_0, t_f):
        repeated_laplacian = self.laplacian.unsqueeze(0).repeat(len(xs), 1)

        laplacian_exp = exp((t_f - t_0) * self.diffusion * self.laplacian)
        repeated_laplacian_exp = laplacian_exp.unsqueeze(0).repeat(len(xs), 1)

        forcings = (self.forcing_hats.T @ us.T).T

        return repeated_laplacian_exp * xs + (repeated_laplacian_exp - 1) * forcings / (self.diffusion * repeated_laplacian)


# Periodic problem with Fourier transform
class FourierSpectral(Spectral):
    def __init__(self, interval, diffusion, forcings, num_modes=None):
        SystemDynamics.__init__(self, interval.N, len(forcings))
        self.interval = interval
        self.diffusion = diffusion
        self.num_modes = num_modes if num_modes is not None else interval.N

        self.laplacian = interval.fourier_truncate(interval.fourier_spec_laplacian, self.num_modes)
        laplacian_copy = self.laplacian.clone()
        laplacian_copy[0] = 1
        self.laplacian_inverse = 1 / laplacian_copy
        self.laplacian_inverse[0] = 0

        forcing_hats = interval.fourier_transform(forcings)
        self.forcing_hats = interval.fourier_truncate(forcing_hats, self.num_modes)
        self.forcings = interval.inverse_fourier_transform(interval.fourier_extend(self.forcing_hats))

    def step(self, xs, us, t_0, t_f):
        repeated_laplacian_inverse = self.laplacian_inverse.unsqueeze(0).repeat(len(xs), 1)

        laplacian_exp = exp((t_f - t_0) * self.diffusion * self.laplacian)
        repeated_laplacian_exp = laplacian_exp.unsqueeze(0).repeat(len(xs), 1)

        us = complex(us, zeros(us.shape))
        forcings = (self.forcing_hats.T @ us.T).T

        x_nexts = repeated_laplacian_exp * xs + (repeated_laplacian_exp - 1) * forcings * repeated_laplacian_inverse / self.diffusion
        x_nexts[:, 0] = xs[:, 0] + (t_f - t_0) * forcings[:, 0]
        return x_nexts
