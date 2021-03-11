from time import perf_counter
from itertools import product

from memory_profiler import memory_usage
import numpy as np

from src.dl_solver_heat2d import solve_heat_with_dl
from src.dl_solver_poisson import solve_poisson_with_dl
from src.fem_solver_heat2d import solve_heat_with_fem
from src.fem_solver_poisson import solve_poisson_with_fem


def _true_solution_heat(t, x, y):
    return 1 + x * x + 3 * y * y + 1.2 * t


def _true_solution_poisson(x, y):
    return 1 + x + 2 * y


def _eval_solver(func, is_dl, is_heat):
    print('Calculations started.')
    start = perf_counter()
    mem_usage, result = memory_usage((func, (True,)), max_usage=True, retval=True)
    stop = perf_counter()
    print('Calculations finished.')
    print(f'Elapsed time: {stop - start:.4f} sec.')
    print(f'Memory footprint: {mem_usage:.2f} MB.')

    print('Measuring accuracy.')
    # mesh
    Lx, Ly = 1, 1
    Nx, Ny = 9, 9
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    xx, yy = np.meshgrid(x, y)

    # true solution
    u_true = None
    samples = None
    if is_heat:
        T = 2
        samples = list(product([T], x, y))
        u_true = np.array([_true_solution_heat(*s) for s in samples])
        u_true = np.reshape(u_true, (Nx, Ny), 'F')
    else:
        samples = list(product(x, y))
        u_true = np.array([_true_solution_poisson(*s) for s in samples])
        u_true = np.reshape(u_true, (Nx, Ny), 'F')

    assert u_true is not None
    assert samples is not None

    # estimated solution
    u_pred = None
    if is_dl:
        u_pred = result.predict(samples)
        u_pred = np.reshape(np.array(u_pred), (Nx, Ny), 'F')
    else:
        if isinstance(result, list):
            # take last time
            u_pred = np.reshape(result[-1], (Nx, Ny))
        else:
            u_pred = np.reshape(result, (Nx, Ny))

    assert u_pred is not None

    # error
    e = np.abs(u_true - u_pred)
    print(f'Average error: {e.mean()}.')
    print(f'Max error: {e.max()}.')


def run_comparison():
    # FEM
    _eval_solver(solve_heat_with_fem, False, True)
    _eval_solver(solve_poisson_with_fem, False, False)

    # DL
    _eval_solver(solve_heat_with_dl, True, True)
    _eval_solver(solve_poisson_with_dl, True, False)


if __name__ == '__main__':
    run_comparison()
