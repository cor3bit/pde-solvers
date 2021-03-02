from functools import partial

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation


# animation function
def _animate(i, x, y, images1d):

    # z = var[i, :, 0, :].T

    im = images1d[i]

    n_dim = int(np.sqrt(len(im)))

    z = np.reshape(im, (n_dim, n_dim))



    cont = plt.contourf(x, y, z, cmap='jet')
    # plt.colorbar()

    # if (i == 0):
    #     plt.title(r't = %1.2e' % t[i])
    # else:
    #     plt.title(r't = %i' % i)

    return cont


def save_dynamic_contours(images1d, meshsize_x, meshsize_y, n_times):
    assert images1d

    n_dim = int(np.sqrt(len(images1d[0])))

    Lx, Ly = meshsize_x, meshsize_y
    Nx, Ny = n_dim, n_dim
    Nt = n_times

    # Generate grid for plotting
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    x, y = np.meshgrid(x, y)


    fig = plt.figure()
    ax = plt.axes(xlim=(0, Lx), ylim=(0, Ly))
    plt.xlabel(r'x')
    plt.ylabel(r'y')

    anim = animation.FuncAnimation(fig, func=partial(_animate, x=x, y=y, images1d=images1d), frames=Nt)

    anim.save('../artifacts/animation.gif', writer='imagemagick')
