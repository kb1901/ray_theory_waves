#%% import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from CartWaveSol import CartWaveSol
from WavePlot import WavePlot
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
bathymetry = gaussian_ridge(xx, yy, 0, 0, 200, 200, 20, 10)
dt = 1
T = 200
E0 = 1e-4
k0 = (1e-3, 0)

wave_sol = CartWaveSol(
    init_k=k0,
    x = x,
    y = y,
    bathymetry = bathymetry,
    x0 = -750,
    y0 = 150,
    E0 = E0,
    T = T,
    dt=dt,
    method='euler'
)

wave_sol.solve()

wave_sol2 = CartWaveSol(
    init_k=k0,
    x = x,
    y = y,
    bathymetry = bathymetry,
    x0 = -750,
    y0 = -150,
    E0 = E0,
    T = T,
    dt=dt,
    method='adams-bashforth'
)

wave_sol2.solve()

# wave_sol3 = CartWaveSol(
#     init_k=(1, 0),
#     x = x,
#     y = y,
#     bathymetry = bathymetry,
#     x0 = -75,
#     y0 = -10,
#     E0 = 1,
#     T = 100,
#     dt=0.1,
#     vegetation=None
# )

# wave_sol3.solve()

#%% plotting

# just a helper
def map_range(src, new_min, new_max):
    src_min = np.min(src)
    src_max = np.max(src)
    output = (
        ((src - src_min) / (src_max - src_min)) * 
        (new_max - new_min) + new_min
    )
    return output

def arrow(x,y,ax,n):
    d = len(x)//(n+1)    
    ind = np.arange(d,len(x),d)
    for i in ind:
        ar = FancyArrowPatch((x[i-1],y[i-1]),(x[i],y[i]), 
                              arrowstyle='->', mutation_scale=50)
        ax.add_patch(ar)

df = wave_sol.df
df2 = wave_sol2.df
# df3 = wave_sol3.df
fig, ax = plt.subplots(dpi=150, figsize=(10,7))
im = ax.pcolormesh(xx, yy, bathymetry, cmap='Reds')
fig.colorbar(im, label='Bathmetry (m)')
s1 = df['A'].to_numpy()/np.max(df['A'].to_numpy())
s2 = df2['A'].to_numpy()/np.max(df2['A'].to_numpy())
arrow(df['x'], df['y'], ax, 5)
arrow(df2['x'], df2['y'], ax, 5)
ax.scatter(df['x'], df['y'], c='aqua', label="Euler's method", edgecolors='black',s=map_range(s1, 10, 80))
ax.scatter(df2['x'], df2['y'], c='orange', label="Adams-Bashforth method", edgecolors='black',s=map_range(s2, 10, 80))
ax.legend(loc='upper right')
ax.xaxis.set_tick_params(direction='in')
ax.yaxis.set_tick_params(direction='in')
# ax.set_xlim(25, 50)
# ax.set_ylim(-10, 10)
# plt.savefig(r'./figures/wave_over_shoal_center.png', dpi=300)
plt.show()

#%% WavePlot try
fig, ax = fig, ax = plt.subplots(dpi=150, figsize=(10,7))
w1 = WavePlot(fig, ax, wave_sol)
w2 = WavePlot(fig, ax, wave_sol2)

w1.plot_wave_traj(xx, yy, bathymetry, color='aqua', label='Euler')
w2.plot_wave_traj(xx, yy, bathymetry, color='orange', label='Adams-Bashforth')
plt.show()