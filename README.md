# EddySearch
[![PyPI version](https://badge.fury.io/py/eddysearch.svg)](https://pypi.org/project/eddysearch/) [![Downloads](https://pepy.tech/badge/eddysearch)](https://pepy.tech/project/eddysearch) [![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity) [![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/) [![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-370/) [![Python 3.6](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/) ![Tests](https://github.com/innvariant/eddysearch/workflows/Tests/badge.svg)

EddySearch is a collection of artificial function landscapes and search strategies.
Artificial landscapes have hills and valleys and good search strategies aim to find these extrema points as fast as possible.
Fast usually means that they use as few function evaluations as possible.
You can look at an artificial landscape [visually (see below)](#visuals) or as a function of the form <img src="https://render.githubusercontent.com/render/math?math=f: \mathbb{R}^d\rightarrow\mathbb{R}"> where *d* is the size (or even shape) of the input dimension and it assigns each point in space a value.
Usually *d* = 2 such that we can imagine a three-dimensional space where we search for the point on a plane which has a certain minimal value.

Optimization or search has various names across many scientific disciplines.
Cost, energy or loss is seeked to minimized or maximized -- often with some additional constraints or by changing parameters of a model.
Artificial neural networks most often seek for [estimates to maximize the log-likelihood](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation)

This also gives the possibility to compare optimization methods in artificial settings:
either for the curious mind, for pedagogical purpose or for fair experimental comparison.
You can simply extend it with your own search strategy and see how well it works in these landscapes (objective functions).

**Jump to ..**
- [Installation](#installation)
- [Introduction / Overview](#intro)
- [Visuals](#visuals)


# Installation
- Install from PyPi via poetry: ``poetry install eddysearch``
- Install with pip: ``pip install eddysearch``
- Install latest development version: ``poetry add git+https://github.com/innvariant/eddysearch.git#master``


# Intro
To make visualizations (e.g. 3D Plots or Manim-Videos) easy, objectives define much more information than just the pure function definition.
For example, they contain information about suggested visualization boundaries or their analytical or empirical known extrema.
Search strategies on the other hand provide capabilities to track their search path through space.
So it is easy to follow their search principles.
The main intent is to provide insights into the differences of various search strategies and how they behave in different artifcial landscapes.

## Example: randomly search through Himmelblau's landscape
```python
import numpy as np
from eddysearch.objective import HimmelblauObjective
from eddysearch.strategy import SearchRunner
from eddysearch.search.randomsearch import RandomUniformSearch

objective = HimmelblauObjective()
objective_lower = objective.search_bounds[:,0]
objective_upper = objective.search_bounds[:,1]
strategy = RandomUniformSearch(objective.dims, objective_lower, objective_upper)
search = SearchRunner(objective, strategy)
search.run()

print('Found minimum argument: f(%s) = %s' % (search._minimum_arg, search._minimum_eval))
print('Used num evaluations: %s' % search._num_evaluations)
print('Known minima of objective are: %s' % objective.minima)
dist_min = np.inf
target_closest = None
target_cost = None
for target_cur in objective.minima:
    dist_cur = np.sqrt(np.sum((target_cur[:-1]-search._minimum_arg)**2))
    if dist_cur < dist_min:
        dist_min = dist_cur
        target_closest = target_cur[:-1]
        target_cost = target_cur[-1]
print('Closest target is: %s with a cost of %s' % (target_closest, target_cost))
```


# Artificial Landscapes
.. also called [test functions for optimization on Wikipedia](https://en.wikipedia.org/wiki/Test_functions_for_optimization).


### Himmelblau Function
Also see [Wikipedia: Himmelblau's function](https://en.wikipedia.org/wiki/Himmelblau%27s_function).

$`f(x,y) = (x^2+y-11(x+ y^2-7)^2)`$

```python
from eddysearch.objective import HimmelblauObjective

obj = HimmelblauObjective()
```



### RastriginObjective

```python
from eddysearch.objective import RastriginObjective

obj = RastriginObjective()
```

### RosenbrockObjective

```python
from eddysearch.objective import RosenbrockObjective

obj = RosenbrockObjective()
```


### LeviN13Objective

```python
from eddysearch.objective import LeviN13Objective

obj = LeviN13Objective()
```


### CrossInTrayObjective

```python
from eddysearch.objective import CrossInTrayObjective

obj = CrossInTrayObjective()
```


### EggholderObjective
```python
from eddysearch.objective import EggholderObjective

obj = EggholderObjective()
```

### Under Development
* Stier2020A1Objective
```python
from eddysearch.objective import Stier2020A1Objective

obj = Stier2020A1Objective()
```

* Stier2020A2Objective
```python
from eddysearch.objective import Stier2020A2Objective

obj = Stier2020A2Objective()
```

* Stier2020BObjective
```python
from eddysearch.objective import Stier2020BObjective

obj = Stier2020BObjective()
```



# Visuals
![Random Search over Himmelblau objective](res/himmelblau-random.png)
![CMA-ES Search over Himmelblau objective](res/himmelblau-cmaes.png)
![Adam Gradient Descent over Himmelblau objective](res/himmelblau-adam.png)
![Random Search over Rastrigin objective](res/rastrigin-random.png)
