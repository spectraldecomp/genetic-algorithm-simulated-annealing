import numpy as np

# Selects the cooling schedule to use
def cooling_schedule(type, T_initial, alpha, t):
    if type == 1:
        return linear_cooling_schedule(T_initial, alpha, t)
    if type == 2:
         return exponential_cooling_schedule(T_initial, alpha, t)
    if type == 3:
         return logarithmic_cooling_schedule(T_initial, alpha, t)
    if type == 4:
         return geometric_cooling_schedule(T_initial, alpha, t)
    if type == 5:
         return power_cooling_schedule(T_initial, alpha, t)
    if type == 5:
         return boltzmann_cooling_schedule(T_initial, t)
    if type == 6:
         return basic_schedule(T_initial, alpha)
     
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


# The temp decreases much like logarithmic cooling, but now there is a square root; alpha determines this rate
def power_cooling_schedule(T_initial, alpha, t):
    return T_initial / (1 + alpha * t) ** 0.5


# The temperature decreases according to Boltzmann.
def boltzmann_cooling_schedule(T_initial, t):
    return T_initial / np.log(t + 1)

# The temperature decreases geometrically independent of time.
def basic_schedule(T_initial, alpha):
    return T_initial*alpha
