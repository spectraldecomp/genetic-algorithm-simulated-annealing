import ga_eval
import ga_util
import langermann_params
import shekel_params
import numpy as np
import random
from simulated_annealing import simulated_annealing



# Define the initial state and temperature
coords_init = np.array([0, 0])
coords_init[0] = np.random.uniform(0, 100)
coords_init[1] = np.random.uniform(0, 100)
T_initial = 1000
T_min = 1e-3
cooling_type = 4

def run_sphere(alpha, max_iters, type):
    sphere = simulated_annealing(ga_eval.sphere, ga_eval.sphere_c, coords_init, T_initial, T_min, alpha, max_iters, type, True)
    coords = sphere.simulated_annealing_run()
    coords[0] = np.round(coords[0], 2)
    coords[1] = np.round(coords[1], 2)
    title = f"Sphere Plot, max_iters= {max_iters}, mode= {cooling_type}, value = {coords}"
    print(title)
    sphere.plot_data(title)

def run_griew(alpha, max_iters, type):
    griew = simulated_annealing(ga_eval.griew, ga_eval.griew_c, coords_init, T_initial, T_min, alpha, max_iters, type, True)
    coords = griew.simulated_annealing_run()
    coords[0] = np.round(coords[0], 2)
    coords[1] = np.round(coords[1], 2)
    title = f"Griew Plot, max_iters= {max_iters}, mode= {cooling_type}, value = {coords}"
    print(title)
    griew.plot_data(title)

def run_shekel(alpha, max_iters, type):
    shekel = simulated_annealing(ga_eval.shekel, ga_eval.shekel_c, coords_init, T_initial, T_min, alpha, max_iters, type, True)
    coords = shekel.simulated_annealing_run()
    coords[0] = np.round(coords[0], 2)
    coords[1] = np.round(coords[1], 2)
    title = f"Shekel Plot, max_iters= {max_iters}, mode= {cooling_type}, value = {coords}"
    print(title)
    shekel.plot_data(title)

def run_michal(alpha, max_iters, type):
    michal = simulated_annealing(ga_eval.micha, ga_eval.micha_c, coords_init, T_initial, T_min, alpha, max_iters, type, False)
    coords = michal.simulated_annealing_run()
    coords[0] = np.round(coords[0], 2)
    coords[1] = np.round(coords[1], 2)
    title = f"Micha Plot, max_iters= {max_iters}, mode= {cooling_type}, value = {coords}"
    print(title)
    michal.plot_data(title)

def run_lang(alpha, max_iters, type):
    lang = simulated_annealing(ga_eval.langermann, ga_eval.langermann_c, coords_init, T_initial, T_min, alpha, max_iters, type, False)
    coords = lang.simulated_annealing_run()
    coords[0] = np.round(coords[0], 2)
    coords[1] = np.round(coords[1], 2)
    title = f"Lang Plot, max_iters= {max_iters}, mode= {cooling_type}, value = {coords}"
    print(title)
    lang.plot_data(title)

def run_odd(alpha, max_iters, type):
    odd = simulated_annealing(ga_eval.odd_square, ga_eval.odd_square_c, coords_init, T_initial, T_min, alpha, max_iters, type, False)
    coords = odd.simulated_annealing_run()
    coords[0] = np.round(coords[0], 2)
    coords[1] = np.round(coords[1], 2)
    title = f"Odd Plot, max_iters= {max_iters}, mode= {cooling_type}, value = {coords}"
    print(title)
    odd.plot_data(title)

def run_bump(alpha, max_iters, type):
    bump = simulated_annealing(ga_eval.bump, ga_eval.bump_c, coords_init, T_initial, T_min, alpha, max_iters, type, False)
    coords = bump.simulated_annealing_run()
    coords[0] = np.round(coords[0], 2)
    coords[1] = np.round(coords[1], 2)
    title = f"Bump Plot, max_iters= {max_iters}, mode= {cooling_type}, value = {coords}"
    print(title)
    bump.plot_data(title)


# for type_ in range(1, 6):
#     run_sphere(0.99, 100, type_)
#     run_sphere(0.99, 1000, type_)
#     run_sphere(0.99, 10000, type_)
# for type_ in range(1, 6):
#     run_griew(0.99, 100, type_)
#     run_griew(0.99, 1000, type_)
#     run_griew(0.99, 10000, type_)
# for type_ in range(1, 6):
#     run_shekel(0.99, 100, type_)
#     run_shekel(0.99, 1000, type_)
#     run_shekel(0.99, 10000, type_)
# for type_ in range(1, 6):
#     run_michal(0.99, 100, type_)
#     run_michal(0.99, 1000, type_)
#     run_michal(0.99, 10000, type_)
# for type_ in range(1, 6):
#     run_lang(0.99, 100, type_)
#     run_lang(0.99, 1000, type_)
#     run_lang(0.99, 10000, type_)
# for type_ in range(1, 6):
#     run_odd(0.99, 100, type_)
#     run_odd(0.99, 1000, type_)
#     run_odd(0.99, 10000, type_)
# for type_ in range(1, 6):
#     run_bump(0.99, 100, type_)
#     run_bump(0.99, 1000, type_)
#     run_bump(0.99, 10000, type_)

