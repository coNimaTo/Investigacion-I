from math import *

patm        = 1     #
T_amb       = 20    # Celsius   #300 Kelvin
nR          = 1     # n (moles) * R (cte gases)

A           = 2
w           = 1

rho_agua    = 1000 # kg/m^3
viscosidad  = 0.01 # Pa*s
s_sup       = 72.74 / 1000 # N/m

def ps(t, w, d = 0):
    return A * sin(t*w + d)

def step_A(t, a, b, dt):    
    return b*dt + a

def step_B(t, a, b, dt):
    return -3/2 * b*b/a                         \
    + (3/4 * nR * T_amb / (pi*a**3))/rho_agua   \
    - (2*s_sup + 4*viscosidad*b)/(a*rho_agua)   \
    - (patm + ps(t, w))/rho_agua                \

