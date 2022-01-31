# EddySearch
Eddy is a collection of artificial function landscapes and search strategies to find their extrema points.
Most artificial landscapes are in euclidean space and can simply be seen as a landscape of hills and valleys and the goal is to find the lowest or highest point within a given evaluation budget.
This gives the possibility to compare optimization methods -- or often also known as search strategies -- in artificial settings:
either for the curious mind, for pedagogical purpose or for fair experimental comparison.
You can simply extend it with your own search strategy and see how well it works in these landscapes (objective functions).


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
