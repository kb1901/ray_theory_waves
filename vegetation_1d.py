#%% import
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit

#%% Constants and helpers
g = 9.81 # m/s^2

def gaussian_ridge(x, mean_x, sigma_x, mean_depth, ridge_height):
    '''
    Create a Gaussian ridge in the bathymetry.
    
    Parameters
    ----------
    x: x grid
    mean_x: mean position of the ridge
    sigma_x: standard deviation of the ridge
    mean_depth: mean depth of the lake
    ridge_height: height of the ridge above the mean depth
    
    Returns
    -------
    bathymetry: 1D array of bathymetry values
    '''
    return mean_depth - ridge_height * np.exp(-((x - mean_x) ** 2) / (2 * sigma_x ** 2))

def sigma_shallow(k, d):
    return np.sqrt(g * d) * k

def sigma_real(k, d):
    return np.sqrt(g * k * np.tanh(k * d))

def cg_shallow(k, d):
    return sigma_shallow(k, d) / k

def cg_real(k, d):
    return sigma_real(k, d) / k

def vegetation_dissipation(
        bathymetry_func,
        x_now,
        energy,
        kind,
        exponent=1.0,
        gamma=1.0,
        **kwargs,
        ):
    '''
    Returns the vegetation function based on type.

    Parameters
    ----------
    bathymetry: interpolating function for bathymetry
    x_now: ray position
    energy: energy of the wave
    kind: one of 'critical', 'switch'
    exponent: exponent to raise energy to
    gamma: dissipation coefficient
    '''
    if kind == 'critical':
        H_crit = kwargs.get('H_crit', 0.0)
        H_now = bathymetry_func(x_now)
        if H_now > H_crit:
            gamma = 0
        return - gamma * energy ** exponent
    
    elif kind == 'switch':
        x_switch_on = kwargs.get('x_switch_on', 0.0)
        x_switch_off = kwargs.get('x_switch_off', 0.0)
        if (x_now < x_switch_on) or (x_now > x_switch_off):
            gamma = 0
        return -gamma * energy ** exponent

def solve(
        x,
        x0,
        bathymetry,
        k0,
        E0,
        vegetation_kind,
        approx='sw',
        exponent=1.0,
        gamma=1.0,
        T=15.0,
        dt=0.1,
        **kwargs
):
    '''
    Numerical solution of the one-D ray tracing equation with vegetation
    If height is less than H_crit, vegetation is present.

    Parameters
    ----------
    x: x grid
    x0: initial position of the ray
    bathymetry: depth
    k0: initial wavenumber
    E0: initial energy
    vegetation_kind: one of 'critical', 'switch'
    approx: 'sw' for shallow water, 'real' for real dispersion relation
    exponent: exponent to raise energy to
    gamma: dissipation coefficient
    T: period of the wave
    dt: time step

    Returns
    -------
    '''
    # kwargs
    H_crit = kwargs.get('H_crit', 0.0)
    x_switch_on = kwargs.get('x_switch_on', 0.0)
    x_switch_off = kwargs.get('x_switch_off', 0.0)

    if approx == 'sw':
        sigma = sigma_shallow
        cg = cg_shallow
    elif approx == 'real':
        sigma = sigma_real
        cg = cg_real


    ts = np.arange(0, T, dt)
    ray_xs = np.zeros(len(ts))
    ks = np.zeros(len(ts))
    Es = np.zeros(len(ts))
    amps = np.zeros(len(ts))
    ray_xs[0] = x0
    ks[0] = k0
    Es[0] = E0
    amps[0] = np.sqrt(2 * E0) / sigma(k0, bathymetry[np.searchsorted(x, x0, side='right')])


    dx = x[1] - x[0]

    bath_interp = CubicSpline(x, bathymetry)


    for i in range(1, len(ts)):

        if ray_xs[i-1] < x[0] or ray_xs[i-1] > x[-1]:
            # If the ray is out of bounds, stop the simulation
            ray_xs[i:] = np.nan
            ks[i:] = np.nan
            Es[i:] = np.nan
            amps[i:] = np.nan

            df = pd.DataFrame(
                {
                    'time': ts,
                    'x': ray_xs,
                    'k': ks,
                    'E': Es,
                    'A': amps
                }
            )
            print('Ray out of bounds, returning current state.')
            return df

        xi = ray_xs[i - 1] + np.sqrt(g * bath_interp(ray_xs[i-1])) * dt
        ki = ks[i-1] - 0.5 * (sigma(ks[i-1], bath_interp(ray_xs[i-1] + dx)) - sigma(ks[i-1], bath_interp(ray_xs[i-1] - dx))) / (2 * dx) * dt 
        Ei = (Es[i-1] - 
              Es[i-1] * ((cg(ks[i-1], bath_interp(ray_xs[i-1] + dx)) - cg(ks[i-1], bath_interp(ray_xs[i-1] - dx))) / (2 * dx)) * dt +
              vegetation_dissipation(bath_interp, ray_xs[i-1], Es[i-1], vegetation_kind, exponent, gamma, **kwargs) * dt
        )
        amp_i = np.sqrt(2 * Ei) / sigma(ki, bath_interp(xi))


        ray_xs[i] = xi
        ks[i] = ki
        Es[i] = Ei
        amps[i] = amp_i
    
    df = pd.DataFrame(
        {
            'time': ts,
            'x': ray_xs,
            'k': ks,
            'E': Es,
            'A': amps
        }
    )

    return df
        

#%% solve toy problem

x = np.linspace(0, 100, 1000)
x0 = 0.5
# bathymetry = gaussian_ridge(
#     x, 
#     mean_x=10, 
#     sigma_x=2, 
#     mean_depth=5, 
#     ridge_height=2
#     )
bathymetry = np.full_like(x, 2.0)
k0 = 20
E0 = 1.0
T = 15
dt = 0.1
exponent=1.5
ts = np.arange(0, T, dt)
approx = 'real'

# if vegetation is 'switch'
x_switch_on = 0.5
x_switch_off = 100.0


df_sw = solve(
    x=x, 
    x0=x0, 
    bathymetry=bathymetry, 
    k0=k0, 
    E0=E0,
    vegetation_kind='switch',
    approx='sw',
    T=T, 
    dt=dt,
    exponent=exponent,
    x_switch_on=x_switch_on,
    x_switch_off=x_switch_off,
    )

df_real = solve(
    x=x, 
    x0=x0, 
    bathymetry=bathymetry, 
    k0=k0, 
    E0=E0,
    vegetation_kind='switch',
    approx='real',
    T=T, 
    dt=dt,
    exponent=exponent,
    x_switch_on=x_switch_on,
    x_switch_off=x_switch_off,
    )
#%% Plot stuff

fig, ax = plt.subplots()
ax.plot(x, -bathymetry)
ax.set_title('Bathymetry')
ax.set_xlabel('x (m)')
ax.set_ylabel('Depth (m)')
ax.set_ylim(-np.max(bathymetry) * 1.10, 0)
plt.show()

fig, ax = plt.subplots()
ax.plot(df_sw['x'], df_sw['k'], label='Shallow Water')
ax.plot(df_real['x'], df_real['k'], label='True Dispersion')
ax.set_title('Wavenumber')
ax.set_xlabel('x (m)')
ax.set_ylabel('Wavenumber')
ax.legend()
plt.show() 

fig, ax = plt.subplots()
ax.plot(df_sw['x'], df_sw['A'], label='Shallow Water')
ax.plot(df_real['x'], df_real['A'], label='True Dispersion' )
ax.set_title('Amplitude')
ax.set_xlabel('x (m)')
ax.set_ylabel('Amplitude (m)')
plt.show()
# %%
def func(x, a, b):
    return a / (x + b)
popt, pcov = curve_fit(func, df_sw['x'].to_numpy(), df_sw['A'].to_numpy(), nan_policy='omit')
amps_fit = func(df_sw['x'], *popt)
plt.figure()
plt.plot(df_sw['x'], df_sw['A'], label='true')
plt.plot(df_sw['x'], amps_fit, label='fit')
plt.title(f'Energy dissipation is proportional to $E^{{{exponent}}}$') 
plt.ylabel('Amplitude (m)')
plt.xlabel('x (m)')
plt.legend()
plt.show()
#%% 
