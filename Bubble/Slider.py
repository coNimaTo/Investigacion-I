import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.widgets import Button, Slider
from scipy.integrate import solve_ivp

matplotlib.rcParams['axes.labelsize'] = 15
variables = [r"$\dot R$ [m/s]", r"$R$ [m]", r"$Temp$ [K]"]

# Constantes
patm        = 98000     # Pascals
T_amb       = 293       # Kelvin
N           = 6e15      # Num Particulas
kb          = 1.38e-23  # J/K

rho_agua    = 1000      # kg/m^3
viscosidad  = 0.001      # Pa*s == kg/s
s_sup       = 7e-2  # N/m

# P acustica
def ps(t, A, w, d = 0):
    return A * np.sin(t*w + d)   

A  = patm
w  = 2*np.pi * 5000
d  = 3*np.pi/2
ps_vars = [A,w,d]

# Sist Eq
def fun(t, y, ps_vars):
    vel, radi, temp = y
    vDot = (
                - 3/2 * np.power(vel,2)
                + (
                    3/4 * N*kb*temp/(np.pi*np.power(radi,3))
                    - (2*s_sup+4*viscosidad*vel)/radi
                    - patm
                    - ps(t, *ps_vars)
                )/rho_agua
            )/radi
    rDot = vel
    tDot = - 2 * temp * vel/radi

    return vDot,rDot,tDot  

# Cond Iniciales
y0 = [0, 1e-3, T_amb] # m/s , m, Kelvin

# DiffEq Solver
def sol(y0, t, args):
    return solve_ivp(fun, t, y0, args = args, max_step = 1e-6, rtol = 1e-12)

t = (0.,0.0015)

# Create the figure and the line that we will manipulate
fig, ax = plt.subplots(2, sharex = True)

ax[-1].set(xlim = t, xlabel = 'Time [s]')
ax[0].set(ylim = (0,0.005))
ax[1].set(ylim = (100,1e6), yscale = 'log')

s = sol(y0, t, args = [[A,w,d]])
lineR, = ax[0].plot(s.t, s.y[1], lw=2)
lineT, = ax[1].plot(s.t, s.y[2], lw=2)


# adjust the main plot to make room for the sliders
fig.subplots_adjust(left=0.25, bottom=0.25)

# Make a horizontal slider to control the frequency.
axfreq = fig.add_axes([0.25, 0.1, 0.65, 0.03])
freq_slider = Slider(
    ax=axfreq,
    label = 'Frequency [Hz]',
    valmin = 2*np.pi * 100,
    valmax = 2*np.pi * 10000,
    valinit = w,
)

# Make a vertically oriented slider to control the amplitude
axamp = fig.add_axes([0.1, 0.25, 0.0225, 0.63])
amp_slider = Slider(
    ax=axamp,
    label = "Amplitude",
    valmin = 0,
    valmax = 10*patm,
    valinit = A,
    orientation="vertical"
)


# The function to be called anytime a slider's value changes
def update(val):
    s = sol(y0, t, args = [[amp_slider.val,freq_slider.val,d]])

    lineR.set_xdata(s.t)
    lineR.set_ydata(s.y[1])
    lineT.set_xdata(s.t)
    lineT.set_ydata(s.y[2])

    fig.canvas.draw_idle()


# register the update function with each slider
freq_slider.on_changed(update)
amp_slider.on_changed(update)

# Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
resetax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', hovercolor='0.975')


def reset(event):
    freq_slider.reset()
    amp_slider.reset()
button.on_clicked(reset)

plt.show()