#%% import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from wave_classes.CartWaveSol import CartWaveSol
from wave_classes.WavePlot import WavePlot
from matplotlib.patches import FancyArrowPatch
# %matplotlib widget


#%% set up bathymetry
x = np.linspace(-1000, 1000, 1000, endpoint=True)
y = np.linspace(-1000, 1000, 1000, endpoint=True)
xx, yy = np.meshgrid(x, y)

def shoal(x, start, stop, initial_depth, final_depth):
    slope = (initial_depth - final_depth) / (stop - start)
    shoal = - ( - initial_depth +
             np.heaviside(x - start, 0) * slope * (x - start) * np.heaviside(stop - x, 0) + 
             np.heaviside(x - stop, slope * (stop - start)) * slope * (stop - start))
    return shoal

def gaussian_ridge(xx, yy, mean_x, mean_y, sigma_x, sigma_y, mean_depth, ridge_height):
    return mean_depth - ridge_height * np.exp(-((xx - mean_x) ** 2 / (2 * sigma_x ** 2) + (yy - mean_y) ** 2 / (2 * sigma_y ** 2)))

# bathymetry = shoal(xx, 0, 30, 20, 0)
bathymetry = gaussian_ridge(xx, yy, 0, 0, 300, 300, 20, 10)
dt = 1
T = 200
E0 = 1e-4
k0 = (1e-3, 0)
y0s = np.arange(-500, 501, 150)
sols = []

for y0 in y0s:
    wave_sol = CartWaveSol(
        init_k=k0,
        x = x,
        y = y,
        bathymetry = bathymetry,
        x0 = -750,
        y0 = y0,
        E0 = E0,
        T = T,
        dt=dt,
        method='adams-bashforth'
    )

    wave_sol.solve()
    sols.append(wave_sol)

#%% plotting
fig, ax = fig, ax = plt.subplots(dpi=150, figsize=(10,7))
for sol in sols:
    wplot = WavePlot(fig, ax, sols)
    wplot.plot_wave_traj(xx, yy, bathymetry, color='aqua')
plt.show()