import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from .objective import Objective


def visualize_objective(objective: Objective, ax=None, colormap_name='twilight_shifted', max_points_per_dimension=30):
    vis_bounds = objective.visualization_bounds

    spaces = [np.arange(bound[0], bound[1], abs(bound[1]-bound[0])/max_points_per_dimension) for bound in vis_bounds]
    grid = np.array(np.meshgrid(*spaces))

    x, y = grid

    xy = grid.reshape(2, -1).T
    z = objective(xy).reshape(x.shape)

    if ax is None:
        fig = plt.figure()
        ax = fig.gca(projection='3d')

    cmap = plt.get_cmap(colormap_name)
    ax.plot_trisurf(
        x.flatten(), y.flatten(), z.flatten(),
        alpha=0.2, cmap=cmap, linewidth=0.08, antialiased=True, norm=objective.color_normalizer
    )

    return ax
