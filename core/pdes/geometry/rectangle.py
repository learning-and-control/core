from numpy import pi
from torch import arange, cat, linspace, meshgrid, stack, sum, zeros
from torch.fft import fft2, ifft2


class Rectangle:
    def __init__(self, Ls, Ns):
        self.Ls = Ls
        self.Ns = Ns
        self.meshes = [linspace(0, L, N + 1) for L, N in zip(Ls, Ns)]
        self.grids = meshgrid(*self.meshes)
        dx, dy = Ls / Ns
        omegas = 2 * pi / Ls

        cyclers = [cat([arange(N // 2, N), arange(N // 2)]) for N in Ns]
        fourier_wave_numbers = [arange(-N // 2, N // 2)[cycler] for N, cycler in zip(Ns, cyclers)]
        fourier_freqs = [omega * wave_numbers for omega, wave_numbers in zip(omegas, fourier_wave_numbers)]
        repeated_fourier_freqs = [fourier_freqs[0].unsqueeze(1).repeat(1, Ns[1]), fourier_freqs[1].unsqueeze(0).repeat(Ns[0], 1)]
        self.fourier_spec_derivatives = stack([1j * freqs for freqs in repeated_fourier_freqs])

        fourier_freq_grids = meshgrid(*fourier_freqs)
        self.fourier_spec_laplacian = -sum(stack(fourier_freq_grids) ** 2, dim=0)

        sine_wave_numbers = [arange(1, N) for N in Ns]
        sine_freqs = [omega * wave_numbers / 2 for omega, wave_numbers in zip(omegas, sine_wave_numbers)]
        sine_freq_grids = meshgrid(*sine_freqs)
        self.sine_spec_laplacian = -sum(stack(sine_freq_grids) ** 2, dim=0)

    def fourier_transform(self, signals):
        signals = signals[:, :self.Ns[0], :self.Ns[1]]
        return fft2(signals)

    def inverse_fourier_transform(self, signal_hats):
        return ifft2(signal_hats).real

    def fourier_truncate(self, signal_hats, num_modes=None):
        if num_modes is None:
            num_modes = self.Ns

        selectors = [cat([arange(num // 2), arange(N - (num // 2), N)]) for num, N in zip(num_modes, self.Ns)]
        return signal_hats[..., selectors[0], :][..., selectors[1]]

    def fourier_extend(self, signal_hats):
        shape = signal_hats.shape
        num_modes = shape[-2:]
        selectors = [cat([arange(num // 2), arange(N - (num // 2), N)]) for num, N in zip(num_modes, self.Ns)]

        extension = zeros(*shape[:-2], *self.Ns, dtype=signal_hats.dtype)
        selected_extension = extension[..., selectors[0], :]
        selected_extension[..., selectors[1]] = signal_hats
        extension[..., selectors[0], :] = selected_extension

        return extension

    def sine_transform(self, signals):
        raise NotImplementedError

    def inverse_sine_transform(self, signal_hats):
        raise NotImplementedError

    def sine_truncate(self, signal_hats, num_modes=None):
        if num_modes is None:
            num_modes = self.Ns - 1

        return signal_hats[..., :num_modes[0], :num_modes[1]]

    def sine_extend(self, signal_hats):
        shape = signal_hats.shape
        extension = zeros(*shape[:-2], *(self.Ns - 1))
        extension[..., :shape[-2], :shape[-1]] = signal_hats
        return extension
