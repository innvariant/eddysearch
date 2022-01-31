# EddySearch  [![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity) [![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/) [![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-370/) [![Python 3.6](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/) ![Tests](https://github.com/innvariant/eddysearch/workflows/Tests/badge.svg)
Eddy is a collection of artificial function landscapes and search strategies to find their extrema points.
Most artificial landscapes are in euclidean space and can simply be seen as a landscape of hills and valleys and the goal is to find the lowest or highest point within a given evaluation budget.
This gives the possibility to compare optimization methods -- or often also known as search strategies -- in artificial settings:
either for the curious mind, for pedagogical purpose or for fair experimental comparison.
You can simply extend it with your own search strategy and see how well it works in these landscapes (objective functions).

**Jump to ..**
- [Installation](#installation)
- [Introduction / Overview](#intro)

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

![Random Search over Himmelblau objective](res/himmelblau-random.png)
![CMA-ES Search over Himmelblau objective](res/himmelblau-cmaes.png)
![Adam Gradient Descent over Himmelblau objective](res/himmelblau-adam.png)
![Random Search over Rastrigin objective](res/rastrigin-random.png)



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
