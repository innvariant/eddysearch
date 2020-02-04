import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from .objective import Objective


def visualize_objective(objective: Objective, ax=None, colormap_name='twilight_shifted', max_points_per_dimension=30):
    assert objective.dims == 2

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


def adjust_lightness(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


def obtain_path_centroid(x, objective: Objective=None):
    """

    :param x: d-dimensional points of amount n with shape (d,n)
    :return: Single d-dimensional point (d,)
    """
    assert len(x.shape) == 2
    num_points = x.shape[1]
    return np.sum(x, axis=1)/num_points


def obtain_first_element(x, objective: Objective=None):
    """

    :param x: d-dimensional points of amount n with shape (d,n)
    :return: Single d-dimensional point (d,)
    """
    assert len(x.shape) == 2
    return x[:,0]


def obtain_maximum_objective(x, objective: Objective=None):
    """

    :param x: d-dimensional points of amount n with shape (d,n)
    :return: Single d-dimensional point (d,)
    """
    assert len(x.shape) == 2
    return np.maximum([objective.evaluate_visual(p) for p in x])


fn_path_centroid_selectors = {
    'centroid': obtain_path_centroid,
    'first': obtain_first_element,
    'max': obtain_maximum_objective
}


def visualize_path(search_path, objective: Objective = None, axis=None, connected_path=True, path_centroid_aggregate='centroid'):
    assert hasattr(search_path, 'shape'), 'Visualization of path needs a shape information for accessing dimension sizes'
    assert search_path.shape[0] == 2 or search_path.shape[0] == 3, 'We expect the first dimension of the search path to match the dimensions of the points. Currently only 2d and 3d is supported'

    num_dims = search_path.shape[0]
    num_points = search_path.shape[-1]

    if axis is None:
        fig = plt.figure()
        axis = fig.gca(projection='3d')

    if connected_path:
        fn_path_centroid = None
        if path_centroid_aggregate == 'centroid':
            fn_path_centroid