from numpy import pi
from torch import arange, cat, linspace, ones, sparse_coo_tensor, stack, tensor, zeros
from torch.fft import fft, ifft


class Interval:
    def __init__(self, L, N):
        self.L = L
        self.N = N
        self.mesh = linspace(0, L, N + 1)
        dx = L / N
        omega = 2 * pi / L

        row_indices = cat([arange(N), arange(1, N + 1), tensor([0, N])])
        col_indices = cat([arange(1, N + 1), arange(N), tensor([0, N])])
        values = cat([tensor([2]), ones(N - 1), -ones(N - 1), tensor([-2]), tensor([-2, 2])]) / (2 * dx)
        self.finite_diff_derivative = sparse_coo_tensor(stack([row_indices, col_indices]), values, (N + 1, N + 1))

        row_indices = cat([arange(N - 1), arange(1, N), tensor([0, N - 1])])
        col_indices = cat([arange(1, N), arange(N - 1), tensor([N - 1, 0])])
        values = cat([ones(N - 1), -ones(N - 1), tensor([-1, 1])]) / (2 * dx)
        self.periodic_derivative = sparse_coo_tensor(stack([row_indices, col_indices]), values, (N, N))

        row_indices = cat([arange(N - 2), arange(1, N - 1)])
        col_indices = cat([arange(1, N - 1), arange(N - 2)])
        values = cat([ones(N - 2), -ones(N - 2)]) / (2 * dx)
        self.dirichlet_derivative = sparse_coo_tensor(stack([row_indices, col_indices]), values, (N - 1, N - 1))

        row_indices = tensor([0, N - 2])
        col_indices = tensor([0, 1])
        values = tensor([-1, 1]) / (2 * dx)
        self.dirichlet_derivative_bcs = sparse_coo_tensor(stack([row_indices, col_indices]), values, (N - 1, 2))

        row_indices = cat([arange(N - 1), arange(N), arange(1, N), tensor([0, N - 1])])
        col_indices = cat([arange(1, N), arange(N), arange(N - 1), tensor([N - 1, 0])])
        values = cat([ones(N - 1), -2 * ones(N), ones(N - 1), tensor([1, 1])]) / (dx ** 2)
        self.periodic_laplacian = sparse_coo_tensor(stack([row_indices, col_indices]), values, (N, N))

        row_indices = cat([arange(N - 2), arange(N - 1), arange(1, N - 1)])
        col_indices = cat([arange(1, N - 1), arange(N - 1), arange(N - 2)])
        values = cat([ones(N - 2), -2 * ones(N - 1), ones(N - 2)]) / (dx ** 2)
        self.dirichlet_laplacian = sparse_coo_tensor(stack([row_indices, col_indices]), values, (N - 1, N - 1))

        row_indices = tensor([0, N - 2])
        col_indices = tensor([0, 1])
        values = tensor([1, 1]) / (dx ** 2)
        self.dirichlet_laplacian_bcs = sparse_coo_tensor(stack([row_indices, col_indices]), values, (N - 1, 2))

        cycler = cat([arange(N // 2, N), arange(N // 2)])
        fourier_wave_numbers = arange(-N // 2, N // 2)[cycler]
        fourier_freqs = omega * fourier_wave_numbers
        self.fourier_spec_derivative = 1j * fourier_freqs
        self.fourier_spec_laplacian = -(fourier_freqs ** 2)

        sine_wave_numbers = arange(1, N)
        sine_freqs = omega * sine_wave_numbers / 2
        self.sine_spec_laplacian = -(sine_freqs ** 2)

    def fourier_transform(self, signals):
        signals = signals[:, :self.N]
        return fft(signals)

    def inverse_fourier_transform(self, signal_hats):
        return ifft(signal_hats).real

    def fourier_truncate(self, signal_hats, num_modes=None):
        if num_modes is None:
            num_modes = self.N

        selector = cat([arange(num_modes // 2), arange(self.N - (num_modes // 2), self.N)])
        return signal_hats[..., selector]

    def fourier_extend(self, signal_hats):
        shape = signal_hats.shape
        num_modes = shape[-1]
        selector = cat([arange(num_modes // 2), arange(self.N - (num_modes // 2), self.N)])

        extension = zeros(*shape[:-1], self.N, dtype=signal_hats.dtype)
        extension[..., selector] = signal_hats
        return extension

    def sine_transform(self, signals):
        signals = signals[:, :self.N]
        extended_signals = cat([signals, zeros(len(signals), 1), -signals[:, 1:].flip(-1)], dim=-1)
        extended_signal_hats = fft(extended_signals)
        return -extended_signal_hats[:, 1:self.N].imag / self.N

    def inverse_sine_transform(self, signal_hats):
        signal_hats = -1j * self.N * signal_hats
        Z = zeros(len(signal_hats), 1)
        extended_signal_hats = cat([Z, signal_hats, Z, -signal_hats.flip(-1)], dim=-1)
        return ifft(extended_signal_hats).real[:, :self.N]

    def sine_truncate(self, signal_hats, num_modes=None):
        if num_modes is None:
            num_modes = self.N - 1

        return signal_hats[..., :num_modes]

    def sine_extend(self, signal_hats):
        shape = signal_hats.shape
        extension = zeros(*shape[:-1], self.N - 1)
        extension[..., :shape[-1]] = signal_hats
        return extension
