"""
Example adapted from
https://github.com/hplgit/fenics-tutorial/blob/master/pub/python/vol1/ft03_heat.py

FEniCS tutorial demo program: Heat equation with Dirichlet conditions.
Test problem is chosen to give an exact solution at all nodes of the mesh.
  u'= Laplace(u) + f  in the unit square
  u = u_D             on the boundary
  u = u_0             at t = 0
  u = 1 + x^2 + alpha*y^2 + \beta*t
  f = beta - 2 - 2*alpha
"""

import numpy as np
import fenics as fs
import matplotlib.pyplot as plt

from src.utils.plotting import save_dynamic_contours


def boundary(x, on_boundary):
    return on_boundary


if __name__ == '__main__':
    T = 2.0  # final time
    num_steps = 100  # number of time steps
    dt = T / num_steps  # time step size
    alpha = 3  # parameter alpha
    beta = 1.2  # parameter beta

    # Create mesh and define function space
    nx = ny = 8
    mesh = fs.UnitSquareMesh(nx, ny)
    V = fs.FunctionSpace(mesh, 'P', 1)

    # Define boundary condition
    u_D = fs.Expression('1 + x[0]*x[0] + alpha*x[1]*x[1] + beta*t',
                        degree=2, alpha=alpha, beta=beta, t=0)

    bc = fs.DirichletBC(V, u_D, boundary)

    # Define initial value
    u_n = fs.interpolate(u_D, V)
    # u_n = project(u_D, V)

    # Define variational problem
    u = fs.TrialFunction(V)
    v = fs.TestFunction(V)
    f = fs.Constant(beta - 2 - 2 * alpha)

    F = u * v * fs.dx + dt * fs.dot(fs.grad(u), fs.grad(v)) * fs.dx - (u_n + dt * f) * v * fs.dx
    a, L = fs.lhs(F), fs.rhs(F)

    # Time-stepping
    u = fs.Function(V)
    t = 0

    images1d = []

    for n in range(num_steps):
        # Update current time
        t += dt
        u_D.t = t

        # Compute solution
        fs.solve(a == L, u, bc)

        # Restore numpy object
        image1d = np.empty((81,), dtype=np.float)
        for v in fs.vertices(mesh):
            image1d[v.index()] = u(*mesh.coordinates()[v.index()])

        images1d.append(image1d)

        # Compute error at vertices
        u_e = fs.interpolate(u_D, V)
        error = np.abs(u_e.vector().get_local() - u.vector().get_local()).max()
        print('t = %.2f: error = %.3g' % (t, error))

        # Update previous solution
        u_n.assign(u)

    # Plotting
    # fs.plot(u)
    # plt.show()

    save_dynamic_contours(images1d, 1.0, 1.0, 'heat2d')
