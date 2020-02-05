import matplotlib.colors
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from eddy.search.geneticsearch import GeneticGridSearch, GeneticRingSearch
from eddy.search.gradient import SGDSearch, MomentumSGDSearch, NesterovMomentumSGDSearch, AdamSGDSearch
from eddy.search.population import CMAESSearch, RandomEvolutionarySearch, SpeciesCMAESSearch
from eddy.search.randomsearch import RandomUniformSearch
from eddy.objective import RastriginObjective, GoldsteinPriceObjective, LeviN13Objective, HimmelblauObjective, \
    EggholderObjective, CrossInTrayObjective, Stier2020A1Objective, Stier2020A2Objective
from eddy.strategy import SearchRunner
from eddy.visualization import visualize_objective, visualize_path


def round_partial(value, resolution):
    return round(value / resolution) * resolution


def visualize_search_path(ax, objective, search_path, color='red'):
    for idx in np.arange(0, len(search_path)):
        linewidth = idx/len(search_path) + 0.1
        alpha = round_partial(idx/len(search_path), 0.1)
        #color = adjust_lightness('red',  1+round((idx/len(search_path)+0.1)*0.4, 2))
        for point in search_path[idx]:
            step_set_x = np.array([point[0]])
            step_set_y = np.array([point[1]])
            step_set_z = np.array([objective(point)])
            ax.plot(step_set_x, step_set_y, step_set_z, alpha=alpha, marker='.', linewidth=linewidth, color=color)


objective = HimmelblauObjective()
objective_lower = objective.search_bounds[:,0]
objective_upper = objective.search_bounds[:,1]

num_repetitions = 2
do_visualize_search_path = True
visualize_objective_lower = objective.visualization_bounds[:,0]
visualize_objective_upper = objective.visualization_bounds[:,1]
visualize_color_normalizer = matplotlib.colors.LogNorm()
#visualize_max_grid_step = 100

strategy_genetic_grid = GeneticGridSearch(
    dimensions=2,
    lower=objective_lower,
    upper=objective_upper,
    population_size=20,
    num_generations=20,
    binary_space=5
)
strategy_genetic_ring = GeneticRingSearch(
    dimensions=2, lower=objective_lower, upper=objective_upper, population_size=20, num_generations=80,
    min_radius=3.1, max_radius=10.0
)
strategy_random_uniform = RandomUniformSearch(2, objective_lower, objective_upper)
gradient_learning_rate = 1/100*(1/np.max(np.abs(np.subtract(objective_lower, objective_upper))))
strategy_sgd = SGDSearch(
    dimensions=2,
    lower=objective_lower,
    upper=objective_upper,
    learning_rate=gradient_learning_rate
)
strategy_momentumsgd = MomentumSGDSearch(
    dimensions=2,
    lower=objective_lower,
    upper=objective_upper,
    momentum=0.9,
    learning_rate=gradient_learning_rate
)
strategy_nesterovsgd = NesterovMomentumSGDSearch(
    dimensions=2,
    lower=objective_lower,
    upper=objective_upper,
    momentum=0.9,
    learning_rate=gradient_learning_rate
)
strategy_adamsgd = AdamSGDSearch(
    dimensions=2,
    lower=objective_lower,
    upper=objective_upper,
    learning_rate=gradient_learning_rate
)
strategy_cmaes = CMAESSearch(
    dimensions=2,
    lower=objective_lower,
    upper=objective_upper,
    population_size=20
)
strategy_speciescmaes = SpeciesCMAESSearch(
    dimensions=2,
    lower=objective_lower,
    upper=objective_upper,
    population_size=20
)
strategy_randomevolutionary = RandomEvolutionarySearch(
    dimensions=2,
    lower=objective_lower,
    upper=objective_upper
)


use_strategy = strategy_speciescmaes

repeated_minimum = []
repeated_args = []
for repetition in range(num_repetitions):
    print('Use strategy: %s' % str(use_strategy))
    search = SearchRunner(objective, use_strategy, soft_evaluation_limit=200)
    search.run()

    print('Found minimum argument: f(%s) = %s' % (search._minimum_arg, search._minimum_eval))
    print('Used num evaluations: %s' % search._num_evaluations)
    repeated_minimum.append(search._minimum_eval)
    repeated_args.append(search._minimum_arg)
    #print(rastrign_search._search_path[:10])

    if do_visualize_search_path:
        ax = visualize_objective(objective, max_points_per_dimension=50, colormap_name='jet')

        visualize_path(search._search_path, objective, axis=ax)
        #ax.set_zlim3d(-1, 30000)
        plt.show()

print('Repeated Search. List of found minima:')
print(repeated_minimum)
for agg in [np.mean, np.max, np.min, np.median]:
    print('%s of minima' % agg.__name__, agg(repeated_minimum))


#ax = sns.boxplot(repeated_minimum)
#ax.set_xlim([-960, -700])
#plt.show()

#errors = [np.minimum([np.abs(found_val, min_val[-1]) for min_val in objective.minima]) for found_val in repeated_minimum]
errors = [np.min([np.abs(found_val-min_val[-1]) for min_val in objective.minima]) for found_val in repeated_minimum]
print('Errors:')
print(errors)
for agg in [np.mean, np.max, np.min, np.median]:
    print('%s of errors' % agg.__name__, agg(errors))
ax = sns.boxplot(errors)
ax.set_xlim([0, 300])
plt.title(str(objective)+'\n'+str(use_strategy))
plt.show()