import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.art3d as art3d
import numpy as np

from matplotlib.patches import Ellipse

from .objective import Objective


def visualize_objective(
    objective: Objective,
    ax=None,
    colormap_name="twilight_shifted",
    max_points_per_dimension=30,
):
    assert objective.dims == 2

    vis_bounds = objective.visualization_bounds

    spaces = [
        np.arange(
            bound[0], bound[1], abs(bound[1] - bound[0]) / max_points_per_dimension
        )
        for bound in vis_bounds
    ]
    grid = np.array(np.meshgrid(*spaces))

    x, y = grid

    xy = grid.reshape(2, -1).T
    z = objective.evaluate_visual(xy).reshape(x.shape)

    if ax is None:
        fig = plt.figure()
        ax = fig.gca(projection="3d")
        ax.set_xlim3d(vis_bounds[0][0], vis_bounds[0][1])
        ax.set_ylim3d(vis_bounds[1][0], vis_bounds[1][1])

    cmap = plt.get_cmap(colormap_name)
    ax.plot_trisurf(
        x.flatten(),
        y.flatten(),
        z.flatten(),
        alpha=0.2,
        cmap=cmap,
        linewidth=0.08,
        antialiased=True,
        norm=objective.color_normalizer,
    )

    return ax


def adjust_lightness(color, amount=0.5):
    import colorsys

    import matplotlib.colors as mc

    try:
        c = mc.cnames[color]
    except IndexError:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


def obtain_path_centroid(x, objective: Objective = None):
    """

    :param x: d-dimensional points of amount n with shape (n,d)
    :return: Single d-dimensional point (d,)
    """
    assert len(x.shape) == 2
    # num_points = x.shape[0]
    # return np.sum(x, axis=0)/num_points
    return np.mean(x, axis=0)


def obtain_first_element(x, objective: Objective = None):
    """

    :param x: d-dimensional points of amount n with shape (d,n)
    :return: Single d-dimensional point (d,)
    """
    assert len(x.shape) == 2
    return x[0]


def obtain_extremum_objective(x, objective: Objective, comp=np.greater):
    """

    :param x: d-dimensional points of amount n with shape (d,n)
    :return: Single d-dimensional point (d,)
    """
    assert len(x.shape) == 2
    extremum_x = x[0]
    extremum_z = objective.evaluate_visual(x[0])
    for p in x[1:]:
        z = objective.evaluate_visual(p)
        if comp(z, extremum_z):
            extremum_x = p
            extremum_z = z
    return extremum_x


def obtain_maximum_objective(x, objective: Objective):
    return obtain_extremum_objective(x, objective)


def obtain_minimum_objective(x, objective: Objective):
    return obtain_extremum_objective(x, objective, comp=np.smaller)


fn_path_centroid_selectors = {
    "centroid": obtain_path_centroid,
    "first": obtain_first_element,
    "max": obtain_maximum_objective,
    "min": obtain_minimum_objective,
}


def old_visualize_path(
    search_path,
    objective: Objective = None,
    space: int = None,
    infer_objective: bool = True,
    axis=None,
    connected_path=True,
    path_centroid_aggregate="centroid",
):
    assert hasattr(
        search_path, "shape"
    ), "Visualization of path needs a shape information for accessing dimension sizes"
    assert (
        search_path.shape[-1] == 2 or search_path.shape[-1] == 3
    ), "We expect the last dimension of the search path to match the dimensions of the points. Currently only 2d and 3d is supported"

    num_dims = search_path.shape[-1]
    # num_points = search_path.shape[-2]

    infer_objective = infer_objective and objective is not None
    if space is None:
        # Space will contain the dimensional space we are in with our points, e.g. 1D or 2D
        if num_dims == 2:
            if not infer_objective:
                space = 1  # we have no objective (or mustnt use it) and only a tuple (x1, x2), so x1 must be the point and x2 the objective
            else:
                space = 2  # we have an objective available, so we can infer a third dimension for the objective. (x1, x2) -> (z)
        elif num_dims == 3:
            space = 2  # we have two points (x1, x2) available and use the third (.,.,x3) as z and might use the objective for extra interpolation

    if axis is None:
        fig = plt.figure()
        axis = fig.gca(projection="3d")

    if connected_path:
        if len(search_path.shape) > 2:
            # We have an extra dimension preceeding (n,d) and call it "g" for generations or populations: (?,g,n,d)
            fn_path_centroid = (
                fn_path_centroid_selectors[path_centroid_aggregate]
                if path_centroid_aggregate in fn_path_centroid_selectors
                else path_centroid_aggregate
            )
            assert fn_path_centroid is not None

            # Compute centroids along the second last axis (num_points) and reduce (?,?,g,n,d) to (?,?,g,d)
            centroids = np.array(
                [
                    fn_path_centroid(g.reshape(-1, g.shape[-1]), objective)
                    for g in search_path.reshape(
                        -1, search_path.shape[-2], search_path.shape[-1]
                    )
                ]
            ).reshape(np.delete(search_path.shape, len(search_path.shape) - 2, 0))

            # Now we want to flatten all dimensions from (?,?,g,d) to (?,g,d) to treat each path of (,d) as grouped paths we will plot together
            grouped_centroid_paths = centroids.reshape(
                -1, centroids.shape[-2], centroids.shape[-1]
            )
            # print('Grouped centroid paths shape', grouped_centroid_paths.shape)
            for group, group_color in zip(grouped_centroid_paths, ["g", "y", "b"]):
                # Each point in group is now d-dimensional
                for ix in range(group.shape[-2] - 1):
                    xs = group[ix : ix + 2, 0]
                    ys = group[ix : ix + 2, 1]
                    zs = [objective.evaluate_visual([px, py]) for px, py in zip(xs, ys)]
                    axis.plot(xs, ys, zs, ls="--", lw=0.8, color=group_color)
                    # axis.scatter(np.concatenate([p1, [objective.evaluate_visual(p1)]]), np.concatenate([p2, [objective.evaluate_visual(p2)]]), marker='>', color=group_color)


def visualize_path(search_paths, objective: Objective = None, axis=None):
    num_species = len(search_paths)
    colormap = plt.cm.get_cmap("hsv", num_species)
    for species_idx, species in enumerate(search_paths):
        group_color = colormap(species_idx)
        species_centroids = []
        for gen_id, generation in enumerate(species):
            gen_centroid, gen_covariance = visualize_group(
                generation,
                axis,
                objective=objective,
                plot_covariance=True,
                plot_pointwise=".",
                group_color=group_color,
                group_size=0.8,
            )
            species_centroids.append(gen_centroid)
        visualize_group(
            np.array(species_centroids),
            axis,
            objective=objective,
            connect_pairwise=True,
            plot_pointwise=">",
            group_color=group_color,
            group_size=1.2,
        )


def visualize_group(
    group,
    axis,
    connect_pairwise=False,
    plot_pointwise: str = None,
    plot_covariance=False,
    objective: Objective = None,
    group_color="b",
    group_size=1.0,
):
    assert hasattr(group, "shape")
    assert len(group.shape) == 2
    assert group.shape[1] == 2 or group.shape[1] == 3

    num_dims = group.shape[1]
    num_points = group.shape[0]

    centroid = np.mean(group, axis=0)
    covariance = np.cov(group.T)

    if plot_pointwise or num_points < 2 * num_dims:
        marker = plot_pointwise if isinstance(plot_pointwise, str) else "."
        xs = group[:, 0]
        ys = group[:, 1]
        if num_dims == 2:
            zs = [objective.evaluate_visual([px, py]) for px, py in zip(xs, ys)]
        else:
            zs = group[:, 2]
        axis.scatter(xs, ys, zs, marker=marker, color=group_color, lw=group_size * 0.8)

    if connect_pairwise:
        for ix in range(num_points - 1):
            xs = group[ix : ix + 2, 0]
            ys = group[ix : ix + 2, 1]
            if num_dims == 2:
                zs = [objective.evaluate_visual([px, py]) for px, py in zip(xs, ys)]
            else:
                zs = group[ix : ix + 2, 2]
            axis.plot(xs, ys, zs, ls="--", lw=group_size * 0.8, color=group_color)

    if plot_covariance and num_points > num_dims * 2:
        n_std = 3.0
        pearson = covariance[0, 1] / np.sqrt(covariance[0, 0] * covariance[1, 1])
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)
        scale_y = np.sqrt(covariance[1, 1]) * n_std
        scale_x = np.sqrt(covariance[0, 0]) * n_std
        ellipse = Ellipse(
            centroid,
            width=ell_radius_x * scale_x,
            height=ell_radius_y * scale_y,
            facecolor=group_color,
            edgecolor=group_color,
            ls="--",
            lw=group_size * 0.8,
            alpha=0.1,
        )
        ellipse_z = objective.evaluate_visual(centroid)

        axis.add_patch(ellipse)
        art3d.pathpatch_2d_to_3d(ellipse, z=ellipse_z, zdir="z")

    return centroid, covariance
