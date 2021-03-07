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

from src.utils.plotting import save_dynamic_contours


def q(u):
    "Return nonlinear coefficient"
    return 1 + u ** 2


# Use SymPy to compute f from the manufactured solution u
import sympy as sym

if __name__ == '__main__':
    x, y = sym.symbols('x[0], x[1]')
    u = 1 + x + 2 * y
    f = - sym.diff(q(u) * sym.diff(u, x), x) - sym.diff(q(u) * sym.diff(u, y), y)
    f = sym.simplify(f)
    u_code = sym.printing.ccode(u)
    f_code = sym.printing.ccode(f)
    print('u =', u_code)
    print('f =', f_code)

    # Create mesh and define function space
    mesh = fs.UnitSquareMesh(8, 8)
    V = fs.FunctionSpace(mesh, 'P', 1)

    # Define boundary condition
    u_D = fs.Expression(u_code, degree=2)


    def boundary(x, on_boundary):
        return on_boundary


    bc = fs.DirichletBC(V, u_D, boundary)

    # Define variational problem
    u = fs.Function(V)  # Note: not TrialFunction!
    v = fs.TestFunction(V)
    f = fs.Expression(f_code, degree=2)
    F = q(u) * fs.dot(fs.grad(u), fs.grad(v)) * fs.dx - f * v * fs.dx

    # Compute solution
    fs.solve(F == 0, u, bc)

    # Plot solution
    fs.plot(u)

    # Compute maximum error at vertices. This computation illustrates
    # an alternative to using compute_vertex_values as in poisson.py.
    u_e = fs.interpolate(u_D, V)
    import numpy as np

    error_max = np.abs(u_e.vector().get_local() - u.vector().get_local()).max()
    print('error_max = ', error_max)

    # Hold plot
    # Plotting
    # fs.plot(u)
    plt.show()

    #save_dynamic_contours(images1d, 1.0, 1.0, 'poisson')
