import numpy as np


def cooling_schedule(typeA, T_initial, alpha, t):
    if typeA == 1: return linear_cooling_schedule(T_initial, alpha, t)
    if typeA == 2: return exponential_cooling_schedule(T_initial, alpha, t)
    if typeA == 3: return logarithmic_cooling_schedule(T_initial, alpha, t)
    if typeA == 4: return geometric_cooling_schedule(T_initial, alpha, t)
    #if typeA == 5: return power_cooling_schedule(T_initial, alpha, t)
    if typeA == 5: return boltzmann_cooling_schedule(T_initial, t)
    #if typeA == 6: return stupid_schedule(T_initial, alpha)
# The temp decreases linearly; alpha determines the slope
def linear_cooling_schedule(T_initial, alpha, t):
    return T_initial - alpha * t


# The temp decreases exponentially over time; alpha determines this rate
def exponential_cooling_schedule(T_initial, alpha, t):
    return T_initial * alpha ** t


# The temp decreases logarithmically over time; alpha determines this rate
def logarithmic_cooling_schedule(T_initial, alpha, t):
    return T_initial / (1 + alpha * t)


# The temp decreases geometrically over time; alpha determines this rate
def geometric_cooling_schedule(T_initial, alpha, t):
    return T_initial * alpha ** (t/10)


# The temp decreases much like logarithmic cooling, but now there is a square root; alpha
# determines this rate -- UNUSED
def power_cooling_schedule(T_initial, alpha, t):
    return T_initial / (1 + alpha * t) ** 0.5


# The temperature decreases according to Boltzmann.
def boltzmann_cooling_schedule(T_initial, t):
    return T_initial / np.log(t + 1)

#UNUSED
def stupid_schedule(T_initial, alpha):
    return T_initial*alpha
