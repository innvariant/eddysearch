import matplotlib.colors
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from mpl_toolkits.mplot3d import Axes3D

from eddy.geneticsearch import GeneticGridSearch, GeneticRingSearch
from eddy.randomsearch import RandomUniformSearch
from eddy.objective import RastriginObjective, EggholderObjective, RosenbrockObjective, HimmelblauObjective
from eddy.strategy import Search
from eddy.visualization import visualize_objective, adjust_lightness


def visualize_search_path(ax, search_path, color='red'):
    for idx in np.arange(0, len(search_path)):
        linewidth = idx/len(search_path) + 0.1
        alpha = idx/len(search_path)
        #color = adjust_lightness('red',  1+round((idx/len(search_path)+0.1)*0.4, 2))
        for point in search_path[idx]:
            step_set_x = np.array([point[0]])
            step_set_y = np.array([point[1]])
            step_set_z = np.array([objective(point)])
            ax.plot(step_set_x, step_set_y, step_set_z, alpha=alpha, marker='o', linewidth=linewidth, color=color)


objective = RastriginObjective()
objective_lower = objective.search_bounds[:,0]
objective_upper = objective.search_bounds[:,1]

num_repetitions = 2
do_visualize_search_path = True
visualize_objective_lower = objective.visualization_bounds[:,0]
visualize_objective_upper = objective.visualization_bounds[:,1]
visualize_color_normalizer = matplotlib.colors.LogNorm()
#visualize_max_grid_step = 100

strategy_genetic_grid = GeneticGridSearch(2, objective_lower, objective_upper, population_size=20, num_generations=20, binary_space=5)
strategy_genetic_ring = GeneticRingSearch(
    dimensions=2, lower=objective_lower, upper=objective_upper, population_size=20, num_generations=20,
    min_radius=3.1, max_radius=10.0
)
strategy_random_uniform = RandomUniformSearch(2, objective_lower, objective_upper)

use_strategy = strategy_genetic_ring

repeated_minimum = []
repeated_args = []
for repetition in range(num_repetitions):
    print('Use strategy: %s' % use_strategy.__class__.__name__)
    rastrign_search = Search(objective, use_strategy, soft_evaluation_limit=200)
    rastrign_search.run()

    print('Found minimum argument: f(%s) = %s' % (rastrign_search._minimum_arg, rastrign_search._minimum_eval))
    print('Used num evaluations: %s' % rastrign_search._num_evaluations)
    repeated_minimum.append(rastrign_search._minimum_eval)
    repeated_args.append(rastrign_search._minimum_arg)
    print(rastrign_search._search_path)

    if do_visualize_search_path:
        ax = visualize_objective(objective, max_points_per_dimension=50)
        visualize_search_path(ax, rastrign_search._search_path)
        plt.show()

print('Repeated Search. List of found minima:')
print(repeated_minimum)


ax = sns.boxplot(repeated_minimum)
ax.set_xlim([-960, -700])
plt.show()

#errors = [np.minimum([np.abs(found_val, min_val[-1]) for min_val in objective.minima]) for found_val in repeated_minimum]
errors = [np.min([np.abs(found_val-min_val[-1]) for min_val in objective.minima]) for found_val in repeated_minimum]
print('Errors:')
print(errors)
ax = sns.boxplot(errors)
ax.set_xlim([0, 300])
plt.show()