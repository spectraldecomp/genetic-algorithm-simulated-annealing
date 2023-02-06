import ga_eval
import ga_util
from genetic_algorithm import genetic_algorithm
import numpy as np


def eval1():
    sphere = genetic_algorithm(100, 2, 5, -5, 0.1, 0.9, ga_eval.sphere, True)
    x = sphere.genetic_algorithm_run(100, 2, 5, 100, 0.15, 50)
    print("Sphere min", x)
def eval2():
    griew = genetic_algorithm(100, 2, 200, 0, 0.1, 0.9, ga_eval.griew, True)
    x = griew.genetic_algorithm_run(100, 2, 5, 100, 0.15, 50)
    print("Griew min", x)
def eval3():
    shekel = genetic_algorithm(100, 2, 10, 0, 0.1, 0.9, ga_eval.shekel, True)
    x = shekel.genetic_algorithm_run(100, 2, 5, 100, 0.15, 1500)
    print("Shekel min", x)
    print(ga_eval.shekel(x))
def eval4():
    micha = genetic_algorithm(100, 2, 100, -100, 0.1, 0.9, ga_eval.micha, True)
    x = micha.genetic_algorithm_run(100, 2, 5, 100, 0.15, 50)
    print("Micha min", x)
    print(ga_eval.micha(x))
def eval5():
    langer = genetic_algorithm(100, 2, 10, 0, 0.1, 0.9, ga_eval.langermann, False)
    x = langer.genetic_algorithm_run(100, 2, 5, 100, 0.15, 50)
    print("Langer max", x)
    print(ga_eval.langermann(x))
def eval6():
    odd = genetic_algorithm(100, 2, 5*np.pi, -5*np.pi, 0.1, 0.9, ga_eval.odd_square, False)
    x = odd.genetic_algorithm_run(100, 2, 5, 100, 0.15, 50)
    print("Odd max", x)
    print(ga_eval.odd_square(x))


eval1()
eval2()
eval3()
eval4()
eval5()
eval6()
