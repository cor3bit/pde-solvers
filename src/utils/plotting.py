from itertools import product

import numpy as np
from matplotlib import pyplot as plt
from celluloid import Camera


def save_dynamic_contours(images1d, meshsize_x, meshsize_y, name):
    assert images1d

    n_dim = int(np.sqrt(len(images1d[0])))
    Lx, Ly = meshsize_x, meshsize_y
    Nx, Ny = n_dim, n_dim

    # x, y
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    x, y = np.meshgrid(x, y)

    # z
    fig = plt.figure()
    ax = plt.axes(xlim=(0, Lx), ylim=(0, Ly))
    plt.xlabel(r'x')
    plt.ylabel(r'y')

    camera = Camera(fig)

    for im in images1d:
        z = np.reshape(im, (Nx, Ny))

        cont = plt.contourf(x, y, z, 25, cmap='jet')

        camera.snap()

    animation = camera.animate(interval=200, repeat=False, blit=True)

    # animation.save('../artifacts/animation.mp4')
    animation.save(f'../artifacts/animation_fem_{name}.gif', writer='imagemagick')


def save_from_model(model, meshsize_x, meshsize_y, T, Nt, name):
    dt = T / Nt
    Lx, Ly = meshsize_x, meshsize_y
    Nx, Ny = 9, 9

    # x, y
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    xx, yy = np.meshgrid(x, y)

    # init matplotlib figure
    fig = plt.figure()
    ax = plt.axes(xlim=(0, Lx), ylim=(0, Ly))
    plt.xlabel(r'x')
    plt.ylabel(r'y')

    camera = Camera(fig)

    t = 0
    for _ in range(Nt):
        t += dt

        # TODO
        samples = list(product([t], x, y))
        assert len(samples) == Nx * Ny

        u = model.predict(samples)

        u = np.reshape(np.array(u), (Nx, Ny), 'F')

        cont = plt.contourf(xx, yy, u, 25, cmap='jet')

        camera.snap()

    animation = camera.animate(interval=200, repeat=False, blit=True)

    animation.save(f'../artifacts/animation_dl_{name}.gif', writer='imagemagick')
