import numpy as np
import deepxde as dde
from deepxde.backend import tf

from src.utils.plotting import save_dynamic_contours_from_model


def _pde(x, u):
    du_t = dde.grad.jacobian(u, x, i=0, j=0)
    du_xx = dde.grad.hessian(u, x, i=1, j=1)
    du_yy = dde.grad.hessian(u, x, i=2, j=2)

    return du_t - du_xx - du_yy + 6.8


def _func(x):
    # 1 + x^2 + alpha*y^2 + beta*t
    # alpha = 3
    # beta = 1.2
    return 1. + x[:, 1:2] * x[:, 1:2] + 3 * x[:, 2:] * x[:, 2:] + 1.2 * x[:, 0:1]


def solve_heat_with_dl(lightweight=False):
    geom = dde.geometry.Rectangle(xmin=[0, 0], xmax=[1, 1])

    timedomain = dde.geometry.TimeDomain(0, 2)

    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    bc = dde.DirichletBC(geomtime, _func, lambda _, on_boundary: on_boundary)

    ic = dde.IC(geomtime, _func, lambda _, on_initial: on_initial)

    data = dde.data.TimePDE(
        geomtime,
        _pde,
        ic_bcs=[ic, bc],
        num_domain=400,
        num_boundary=200,
        num_initial=100,
        num_test=10000,
        solution=_func,
    )

    # inputs (2D Heat Eqn) - t, x, y
    # output (solution) - u(t, x, y)
    layer_size = [3] + [32] + [64] + [256] + [64] + [32] + [1]
    activation = 'tanh'
    initializer = 'Glorot uniform'
    net = dde.maps.FNN(layer_size, activation, initializer)
    model = dde.Model(data, net)

    # train model
    model.compile('adam', lr=0.001, metrics=['l2 relative error'])

    losshistory, train_state = model.train(epochs=10000)

    if not lightweight:
        dde.saveplot(losshistory, train_state, issave=True, isplot=True)
        save_dynamic_contours_from_model(model, 1, 1, 2, 100, 'heat2d')

    return model


if __name__ == "__main__":
    solve_heat_with_dl()
