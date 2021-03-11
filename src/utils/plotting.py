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

    n_ims = len(images1d)
    for i, im in enumerate(images1d):
        u = np.reshape(im, (Nx, Ny))
        cont = plt.contourf(x, y, u, 25, cmap='jet')

        # save first and last figure
        if i == 0:
            plt.savefig(f'../artifacts/fem_{name}_t0_02.png')
        if i == n_ims - 1:
            plt.savefig(f'../artifacts/fem_{name}_t2.png')

        camera.snap()

    animation = camera.animate(interval=200, repeat=False, blit=True)

    # animation.save('../artifacts/animation.mp4')
    animation.save(f'../artifacts/animation_fem_{name}.gif', writer='imagemagick')


def save_contour(image1d, meshsize_x, meshsize_y, name):
    # x, y
    Lx, Ly = meshsize_x, meshsize_y
    Nx, Ny = 9, 9
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    xx, yy = np.meshgrid(x, y)

    # u
    samples = list(product(x, y))
    assert len(samples) == Nx * Ny

    u = np.reshape(image1d, (Nx, Ny))

    # chart
    cont = plt.contourf(xx, yy, u, 25, cmap='jet')

    plt.savefig(f'../artifacts/fem_{name}.png')


def save_dynamic_contours_from_model(model, meshsize_x, meshsize_y, T, Nt, name):
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

        samples = list(product([t], x, y))
        assert len(samples) == Nx * Ny

        u = model.predict(samples)
        u = np.reshape(np.array(u), (Nx, Ny), 'F')

        cont = plt.contourf(xx, yy, u, 25, cmap='jet')
        # save first and last figure
        if t == np.isclose(t, 0.02):
            plt.savefig(f'../artifacts/dl_{name}_t0_02.png')
        elif np.isclose(t, 2.):
            plt.savefig(f'../artifacts/dl_{name}_t2.png')

        camera.snap()

    animation = camera.animate(interval=200, repeat=False, blit=True)

    animation.save(f'../artifacts/animation_dl_{name}.gif', writer='imagemagick')


def save_contour_from_model(model, meshsize_x, meshsize_y, name):
    # x, y
    Lx, Ly = meshsize_x, meshsize_y
    Nx, Ny = 9, 9
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    xx, yy = np.meshgrid(x, y)

    # u
    samples = list(product(x, y))
    assert len(samples) == Nx * Ny

    u = model.predict(samples)
    u = np.reshape(np.array(u), (Nx, Ny), 'F')

    # chart
    cont = plt.contourf(xx, yy, u, 25, cmap='jet')

    plt.savefig(f'../artifacts/dl_{name}.png')
