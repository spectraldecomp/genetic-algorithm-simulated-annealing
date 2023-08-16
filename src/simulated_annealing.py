import src.util.ga_eval as ga_eval
import src.util.ga_util as ga_util
import src.util.langermann_params as langermann_params
import src.util.shekel_params as shekel_params
import numpy as np
import random
import matplotlib.pyplot as plt
import src.util.cooling_schedules as cooling_schedules


class simulated_annealing:

    def __init__(self, f, f_constraint, coords_initial, T_initial, T_min, alpha, max_iters, type, is_min):
        self.f = f # function to be optimized
        self.f_constraint = f_constraint # function to check if perturbation is valid
        self.coords_initial = coords_initial # initial coordinates
        self.T_initial = T_initial # initial temperature
        self.T_min = T_min # minimum temperature
        self.alpha = alpha # cooling rate
        self.max_iters = max_iters # maximum number of iterations
        self.coords_history = [] # history of coordinates
        self.type = type # type of cooling schedule
        self.is_min = is_min

    def simulated_annealing_run(self):
        coords = self.coords_initial
        T = self.T_initial
        coords_valid = False
        for i in range(self.max_iters):
            # Generate a random perturbation, and check if perturbation valid. Regenerate until perturbation is in constraints of problem.
            coords_new = coords
            while not coords_valid:
                coords_new = coords + np.random.uniform(-1, 1, size=coords.shape)
                coords_valid = self.f_constraint(coords_new)
            coords_valid = False
            # calculates delta E based on max/min problem
            if (self.is_min):
                delta_E = self.f(coords_new) - self.f(coords)
            else:
                delta_E = self.f(coords) - self.f(coords_new)

            probability = np.exp(-delta_E / T)
            if delta_E < 0 or np.random.uniform(0, 1) < probability:
                coords = coords_new
                self.coords_history.append(coords)
            # adjust temperature
            T = cooling_schedules.cooling_schedule(self.type, self.T_initial, self.alpha, i)

            if T < self.T_min:
                break
        return coords

    def plot_data(self, title):
        x = [point[0] for point in self.coords_history]
        y = [point[1] for point in self.coords_history]
        colors = np.arange(len(self.coords_history))
        plt.scatter(x, y, c=colors, cmap='viridis')
        cbar = plt.colorbar()
        cbar.ax.set_yticklabels(['Less recent', ' ', ' ', ' ', 'More recent'])
        plt.title(title)
        name = f"Plot mode= {self.typeA} alpha = {self.alpha} max_iters= {self.max_iters}.png"
        plt.savefig("plots/" + name)
        plt.show()


