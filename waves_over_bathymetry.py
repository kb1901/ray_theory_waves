#%% import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from CartWaveSol import CartWaveSol
sns.set_style('white')
# %matplotlib widget

#%% set up bathymetry
x = np.linspace(-100, 100, 1000)
y = np.linspace(-100, 100, 1000)
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
bathymetry = gaussian_ridge(xx, yy, 0, 0, 30, 30, 5, 3)

wave_sol = CartWaveSol(
    init_k=(1, 0),
    x = x,
    y = y,
    bathymetry = bathymetry,
    x0 = -75,
    y0 = 20,
    E0 = 1,
    T = 100,
    dt=0.1,
    vegetation=None
)

wave_sol.solve()

wave_sol2 = CartWaveSol(
    init_k=(1, 0),
    x = x,
    y = y,
    bathymetry = bathymetry,
    x0 = -75,
    y0 = -20,
    E0 = 1,
    T = 100,
    dt=0.1,
    vegetation=None
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
df = wave_sol.df
df2 = wave_sol2.df
# df3 = wave_sol3.df
fig, ax = plt.subplots(dpi=150, figsize=(10,7))
im = ax.pcolormesh(xx, yy, bathymetry, cmap='Reds')
fig.colorbar(im, label='Bathmetry (m)')
sns.scatterplot(data=df, x='x', y='y', size='A', sizes=(10, 100))
sns.scatterplot(data=df2, x='x', y='y', size='A', sizes=(10, 100))
# sns.scatterplot(data=df3, x='x', y='y', size='A', sizes=(10, 100))
# ax.scatter(df['x'].iloc[0], df['y'].iloc[0], color='yellow', label='Initial position')
ax.legend()
ax.xaxis.set_tick_params(direction='in')
ax.yaxis.set_tick_params(direction='in')
# ax.set_xlim(25, 50)
# ax.set_ylim(-10, 10)
plt.savefig(r'./figures/wave_over_shoal_center.png', dpi=300)
plt.show()
# %%
