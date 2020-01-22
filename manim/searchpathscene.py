from manimlib.imports import *
from math import cos, sin, pi
from eddy.geneticsearch import GeneticGridSearch, GeneticRingSearch
from eddy.objective import RastriginObjective, EggholderObjective, RosenbrockObjective, HimmelblauObjective
from eddy.strategy import Search

class ObjectiveSurface(ParametricSurface):

    def __init__(self, objective, **kwargs):
        assert objective.dims == 2

        vis_bounds = objective.visualization_bounds
        vis_lower = vis_bounds[:, 0]
        vis_upper = vis_bounds[:, 1]

        kwargs = {
            "u_min": vis_lower[0],
            "u_max": vis_upper[0],
            "v_min": vis_lower[1],
            "v_max": vis_upper[1],
            "checkerboard_colors": [BLUE_D]
        }
        self._objective = objective
        ParametricSurface.__init__(self, self.func, **kwargs)

    def func(self, x, y):
        return np.array([x, y, self._objective.evaluate_visual(np.array([x, y]))])


class Shapes(ThreeDScene):
    def construct(self):
        # Define the original artificial landscape as an objective which we can evaluate
        objective = RastriginObjective()
        objective_lower = objective.search_bounds[:, 0]
        objective_upper = objective.search_bounds[:, 1]
        #objective.visualization_z_shift = -400
        objective.visualization_z_shift = -200

        strategy_genetic_ring = GeneticRingSearch(
            dimensions=2, lower=objective_lower, upper=objective_upper, population_size=20, num_generations=20,
            min_radius=3.1, max_radius=10.0
        )

        use_strategy = strategy_genetic_ring

        # Some text we show at the end of the video
        first_line = TextMobject("Objective Function: %s" % str(objective).replace('_', ''))
        second_line = TextMobject("Search strategy: %s" % str(use_strategy).replace('_', ''))
        second_line.next_to(first_line, DOWN)
        final_line = TextMobject("Thanks for watching.", color=BLUE)

        search = Search(objective, use_strategy, soft_evaluation_limit=200)
        search.run()
        search_path = search._search_path

        #search_path = search_path[:2]

        print('First population group (partly) in search path:')
        print(search_path[0][:5])

        # Given the search path as a list of grouped points (might be populations of points), start creating their
        # math objects as dots
        search_path_mobjects = []
        for group in search_path:
            search_group = []
            for point in group:
                coords = np.append(point, objective.evaluate_visual(point))
                search_group.append(Dot(coords, color=YELLOW))
            search_path_mobjects.append(search_group)

        # Create a surface given the objective function and set up a threedaxes math object
        surface = ObjectiveSurface(objective)
        axes = ThreeDAxes()

        # Set initial position of camera and show creation of objective surface and the three axes
        self.set_camera_orientation(0.1, -PI/3, 300)
        self.play(ShowCreation(axes), ShowCreation(surface))
        self.wait(1)

        # Start ambient motion of camera and add groups of search points
        self.begin_ambient_camera_rotation(rate=0.1)

        display_max_groups = 3
        for group_no, group in enumerate(search_path_mobjects):
            animations = [ShowCreation(mem) for mem in group]

            if group_no >= display_max_groups:
                animations += [FadeOut(mem) for mem in search_path_mobjects[group_no-display_max_groups]]

            self.play(*animations)
            self.wait(0.5)

        self.stop_ambient_camera_rotation()
        self.wait(1)


        # Top-down view on surface
        self.move_camera(0, -PI/2, 120)

        # Remove last search groups
        for group in search_path_mobjects[-display_max_groups:]:
            self.play(*[FadeOut(mem) for mem in group])
            self.wait(0.5)

        # Fade out 3d surface, axes and write final text
        self.wait(1)
        self.play(FadeOut(surface), FadeOut(axes))
        self.play(Write(first_line), Write(second_line))
        self.wait(1)

        # Move camera to origin
        self.move_camera(0, -PI/2)
        self.play(FadeOut(second_line), ReplacementTransform(first_line, final_line))
        self.wait(1)
