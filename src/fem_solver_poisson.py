"""
Example adapted from
https://github.com/hplgit/fenics-tutorial/blob/master/src/vol1/python/poisson_nonlinear.py

FEniCS tutorial demo program: Nonlinear Poisson equation.
  -div(q(u)*grad(u)) = f   in the unit square.
                   u = u_D on the boundary.
"""

import numpy as np
import fenics as fs
import matplotlib.pyplot as plt

from src.utils.plotting import save_contour


def q(u):
    "Return nonlinear coefficient"
    return 1 + u ** 2


def solve_poisson_with_fem(lightweight=False):
    # Create mesh and define function space
    mesh = fs.UnitSquareMesh(8, 8)
    V = fs.FunctionSpace(mesh, 'P', 1)

    # Define boundary condition
    u_code = 'x[0] + 2*x[1] + 1'
    u_D = fs.Expression(u_code, degree=2)

    def boundary(x, on_boundary):
        return on_boundary

    bc = fs.DirichletBC(V, u_D, boundary)

    # Define variational problem
    u = fs.Function(V)  # Note: not TrialFunction!
    v = fs.TestFunction(V)

    # f = fs.Expression(f_code, degree=2)
    f_code = '-10*x[0] - 20*x[1] - 10'
    f = fs.Expression(f_code, degree=2)

    F = q(u) * fs.dot(fs.grad(u), fs.grad(v)) * fs.dx - f * v * fs.dx

    # Compute solution
    fs.solve(F == 0, u, bc)

    # Plot solution
    fs.plot(u)

    # Compute maximum error at vertices. This computation illustrates
    # an alternative to using compute_vertex_values as in poisson.py.
    u_e = fs.interpolate(u_D, V)

    # Restore numpy object
    image1d = np.empty((81,), dtype=np.float)
    for v in fs.vertices(mesh):
        image1d[v.index()] = u(*mesh.coordinates()[v.index()])

    if not lightweight:
        error_max = np.abs(u_e.vector().get_local() - u.vector().get_local()).max()
        print('error_max = ', error_max)

        fs.plot(u)
        plt.show()
        save_contour(image1d, 1.0, 1.0, 'poisson')

    return image1d


if __name__ == '__main__':
    solve_poisson_with_fem()
