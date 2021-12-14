from matplotlib.pyplot import figure
from numpy import arange, array, dot, reshape, zeros
from numpy.linalg import solve
from torch import is_tensor, tensor
from warnings import warn


def arr_map(func, *arr):
    return array(list(map(func, *arr)))


def differentiate(xs, ts, L=3):
    half_L = (L - 1) // 2
    b = zeros(L)
    b[1] = 1

    def diff(xs, ts):
        t_0 = ts[half_L]
        t_diffs = reshape(ts - t_0, (L, 1))
        pows = reshape(arange(L), (1, L))
        A = (t_diffs ** pows).T
        w = solve(A, b)
        return dot(w, xs)

    return array(
        [diff(xs[k - half_L:k + half_L + 1], ts[k - half_L:k + half_L + 1]) for
         k in range(half_L, len(ts) - half_L)])


def torch_guard(params, function):
    torch_params = params
    torch_in = True
    are_tensors = [is_tensor(p) for p in params]
    if not all(are_tensors):
        torch_in = False
        torch_params = [tensor(p) for p in params]
    if not torch_in and any(are_tensors):
        raise TypeError('[ERROR] Inconsistent input types. All numpy or all '
                        'torch.')
    torch_output = function(*torch_params)
    if not torch_in:
        if is_tensor(torch_output):
            return torch_output.detach().numpy()
        else:
            try:
                # assuming output is a collection of tensors
                return [r.detach().numpy() if not r.is_sparse else
                #[WARNING] this is a hack for sparse matrices.
                #If needed change this line to return appropriate sparse matrix
                        r.to_dense().detach().numpy() for r in torch_output]
            except TypeError:
                warn('[WARNING] Using torch guard with unsupported return '
                     'type.')
                return torch_output

    else:
        return torch_output


def default_fig(fig, ax):
    if fig is None:
        fig = figure(figsize=(6, 6), tight_layout=True)

    if ax is None:
        ax = fig.add_subplot(1, 1, 1)

    return fig, ax
