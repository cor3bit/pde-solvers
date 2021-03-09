import numpy as np
import deepxde as dde
from deepxde.backend import tf

from src.utils.plotting import save_contour_from_model


def _pde(x, u):
    du_xx = dde.grad.hessian(u, x, i=0, j=0)
    du_yy = dde.grad.hessian(u, x, i=1, j=1)

    return - du_xx - u * du_yy + 2 - 4 * u


def _boundary(_, on_boundary):
    return on_boundary


def _func(x):
    # 1 + x^2 - 2*y^2
    return 1. + x[:, 0:1] * x[:, 0:1] - 2 * x[:, 1:] * x[:, 1:]


def solve_poisson_with_dl(lightweight=False):
    # geom = dde.geometry.Polygon([[0, 0], [1, 0], [1, -1], [-1, -1], [-1, 1], [0, 1]])
    geom = dde.geometry.Rectangle(xmin=[0, 0], xmax=[1, 1])

    # bc = dde.DirichletBC(geom, lambda x: 0, boundary)
    bc = dde.DirichletBC(geom, _func, lambda _, on_boundary: on_boundary)

    data = dde.data.PDE(
        geom,
        _pde,
        bc,
        num_domain=1200,
        num_boundary=120,
        num_test=10000,
        solution=_func,
    )

    # NN
    activation = 'tanh'
    initializer = 'Glorot uniform'
    net = dde.maps.FNN([2] + [50] * 4 + [1], activation, initializer)
    model = dde.Model(data, net)

    # Train NN
    model.compile('adam', lr=0.001)

    losshistory, train_state = model.train(epochs=10000)

    if not lightweight:
        dde.saveplot(losshistory, train_state, issave=True, isplot=True)
        save_contour_from_model(model, 1, 1, 'poisson')

    return model


if __name__ == "__main__":
    solve_poisson_with_dl()
