from functools import partial

import numpy as np
from matplotlib import pyplot as plt
from celluloid import Camera


def save_dynamic_contours(images1d, meshsize_x, meshsize_y):
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
        z = np.reshape(im, (n_dim, n_dim))

        cont = plt.contourf(x, y, z, cmap='jet')

        camera.snap()

    animation = camera.animate(interval=200, repeat=False, blit=True)

    # animation.save('../artifacts/animation.mp4')
    animation.save('../artifacts/animation.gif', writer='imagemagick')
