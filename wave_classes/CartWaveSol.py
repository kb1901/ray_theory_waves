import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import RegularGridInterpolator as RGI
from warnings import warn
#%% implementation of Johnson's Cartesian solution for wave changes over variable depth (ridge)

class CartWaveSol:

    def __init__(self, 
                 init_k, 
                 x, 
                 y, 
                 bathymetry,
                 x0, 
                 y0, 
                 E0, 
                 T, 
                 dt,
                 dispersion=None,
                 method='euler'
                ):
        
        '''
        Initializes the grid and bathymetry along with initial wave conditions.
        Assumes that the wavefront arrives from the left.

        Parameters
        ----------
        init_k:
            initial wave vector
        x:
            x grid (m), uniformly spaced
        y:
            y grid (m), uniformly spaced
        bathymetry:
            2d array of lake bathymetry (m)
        vegetation:
            2d binary array of where vegetation is present
        x0:
            initial x position of the ray
        y0:
            initial y position of the ray
        E0:
            initial amplitude of the wave
        T:
            simulation time (seconds)
        dt:
            time step size (seconds)

        Returns
        -------
        None
        '''
        
        self.g = 9.81 # m/s^2
        self.init_k = init_k
        self.x0 = x0
        self.y0 = y0
        self.E0 = E0
        self.bathymetry = bathymetry
        self.x, self.y = np.meshgrid(x, y)
        self.x_min, self.x_max = np.min(x), np.max(x)
        self.y_min, self.y_max = np.min(y), np.max(y)
        self.T = T
        self.dt = dt
        if dispersion is None or dispersion == 'shallow':
            self.dispersion = self.sigma_sw
        elif dispersion == 'full':
            self.dispersion = self.sigma_full
        else:
            print("Dispersion must be 'shallow' or 'full'.")
        
        if method == 'euler':
            self.step = self.euler_step
        elif method == 'symplectic_euler':
            self.step = self.symplectic_euler_step
        elif method == 'adams-bashforth':
            self.step = self.adams_bashforth_step
        else:
            raise UserWarning("The method must be one of 'euler', 'symplectic_euler', or 'adams-bashforth'.")

        # derived quantities
        self.ts = np.arange(0, T, dt)
        self.dx = x[1] - x[0]
        self.dy = y[1] - y[0]


        # interpolate
        self.bathymetry_interpolate = RGI((x, y), self.bathymetry.T, method='linear')
 
        # initial amplitude
        d_init = self.bathymetry_interpolate((self.x0, self.y0))
        self.a0 = np.sqrt(2 * self.E0) / self.dispersion(self.init_k[0], self.init_k[1], d_init)
        
        # empty arrays for later
        self.ks = np.zeros(len(self.ts))
        self.ls = np.zeros(len(self.ts))
        self.ray_xs = np.zeros(len(self.ts))
        self.ray_ys = np.zeros(len(self.ts))
        self.energies = np.zeros(len(self.ts))
        self.energies_1 = np.zeros(len(self.ts))
        self.amps = np.zeros(len(self.ts))
        self.df = pd.DataFrame(columns=['time', 'x', 'y', 'k', 'l', 'E', 'A'])


        if self.bathymetry.shape != self.x.shape:
            warn("The shape of the bathymetry must be the same as the gridsize you wish to solve for. Please try again")
            return None

        return None

        
    def sigma_sw(self, k, l, d):
        '''
        Dispersion relation for shallow water
        '''
        return np.sqrt(self.g * d * (k ** 2 + l ** 2))
    
    def sigma_full(self, k, l, d):
        '''
        Full dispersion relation for surface waves
        '''
        k_mag = np.sqrt(k**2 + l**2)
        return np.sqrt(self.g * k_mag * np.tanh(k_mag * d))

        
    def solve(self, symp=False):
        '''
        Solves the wave energy equation and ray tracing equations
        '''

        self.ks[0] = self.init_k[0]
        self.ls[0] = self.init_k[1]
        self.ray_xs[0] = self.x0
        self.ray_ys[0] = self.y0
        self.energies[0] = self.E0
        self.energies[1] = self.E0
        self.amps[0] =  self.a0

        for i in range(1, len(self.ks)):
            # Check if wavefront is out of or close to the domain
            x = self.ray_xs[i-1]
            y = self.ray_ys[i-1]
            if self.bounds_check(x, y):
                break
            
            # calculate bathymetry at current position
            d = self.bathymetry_interpolate((x, y))
            
            # perform the integration step
            # if step does not work, break out of loop
            # typically happens in symplectic method when wavefront
            # runs out of the domain
            try:
                xnow, ynow, know, lnow, Enow = self.step(i)
            except:
                break
            
            # record current step output
            self.ray_xs[i] = xnow
            self.ray_ys[i] = ynow
            self.ks[i] = know
            self.ls[i] = lnow
            self.energies[i] = Enow
            amp_now = np.sqrt(2 * Enow) / (self.dispersion(know, lnow, d))
            self.amps[i] = amp_now
        
        # collect everything in a dataframe at the end
        # truncate dataframe to include points until break
        self.df = pd.DataFrame({
            'time': self.ts,
            'x': self.ray_xs,
            'y': self.ray_ys,
            'k': self.ks,
            'l': self.ls,
            'E': self.energies,
            'A': self.amps
        }).iloc[:i-1] 
            
        return None

    def symplectic_euler_step(self, i):
        '''
        Performs one step of Euler's integration method
        '''
        x = self.ray_xs[i-1]
        y = self.ray_ys[i-1]
        k = self.ks[i-1]
        l = self.ls[i-1]
        E = self.energies[i-1]

        # ray equation
        # dx/dt = cg
        d = self.bathymetry_interpolate((x, y))
        x = x + self.dt * self.cg(k, l, d)[0]
        y = y + self.dt * self.cg(k, l, d)[1]

        # Check if wavefront is out of or close to the domain
        if self.bounds_check(x, y):
            return None

        # wavenumber equation
        # dk/dt = -grad(sigma)
        sigma_x = (self.dispersion(k, l, self.bathymetry_interpolate((x + self.dx, y))) - self.dispersion(k, l, self.bathymetry_interpolate((x - self.dx, y)))) / (2 * self.dx)
        sigma_y = (self.dispersion(k, l, self.bathymetry_interpolate((x, y + self.dy))) - self.dispersion(k, l, self.bathymetry_interpolate((x, y - self.dy)))) / (2 * self.dy)
        k = k - sigma_x * self.dt
        l = l - sigma_y * self.dt

        # energy equation
        # dE/dt = -E div(cg)
        diff_x_cgx = (1 / (2 * self.dx)) * (
                self.cg(k, l, self.bathymetry_interpolate((x + self.dx, y)))[0] - 
                self.cg(k, l, self.bathymetry_interpolate((x - self.dx, y)))[0]
                )
        diff_y_cgy = (1 / (2 * self.dy)) * (
            self.cg(k, l, self.bathymetry_interpolate((x, y + self.dy)))[1] - 
            self.cg(k, l, self.bathymetry_interpolate((x, y - self.dy)))[1]
            )
        E = E - self.dt * E * (diff_x_cgx + diff_y_cgy)

        return x, y, k, l, E

    def euler_step(self, i):
        '''
        Performs one step of Euler's integration method
        '''
        x = self.ray_xs[i-1]
        y = self.ray_ys[i-1]
        k = self.ks[i-1]
        l = self.ls[i-1]
        E = self.energies[i-1]

        # ray equation
        # dx/dt = cg
        d = self.bathymetry_interpolate((x, y))
        xnew = x + self.dt * self.cg(k, l, d)[0]
        ynew = y + self.dt * self.cg(k, l, d)[1]

        # wavenumber equation
        # dk/dt = -grad(sigma)
        sigma_x = (self.dispersion(k, l, self.bathymetry_interpolate((x + self.dx, y))) - self.dispersion(k, l, self.bathymetry_interpolate((x - self.dx, y)))) / (2 * self.dx)
        sigma_y = (self.dispersion(k, l, self.bathymetry_interpolate((x, y + self.dy))) - self.dispersion(k, l, self.bathymetry_interpolate((x, y - self.dy)))) / (2 * self.dy)
        knew = k - sigma_x * self.dt
        lnew = l - sigma_y * self.dt

        # energy equation
        # dE/dt = -E div(cg)
        diff_x_cgx = (1 / (2 * self.dx)) * (
                self.cg(k, l, self.bathymetry_interpolate((x + self.dx, y)))[0] - 
                self.cg(k, l, self.bathymetry_interpolate((x - self.dx, y)))[0]
                )
        diff_y_cgy = (1 / (2 * self.dy)) * (
            self.cg(k, l, self.bathymetry_interpolate((x, y + self.dy)))[1] - 
            self.cg(k, l, self.bathymetry_interpolate((x, y - self.dy)))[1]
            )
        Enew = E - self.dt * E * (diff_x_cgx + diff_y_cgy)

        return xnew, ynew, knew, lnew, Enew
    
    def adams_bashforth_step(self, i):
        '''
        Performs one step of Adams-Bashforth 2
        '''
        x = self.ray_xs[i-1]
        y = self.ray_ys[i-1]
        k = self.ks[i-1]
        l = self.ls[i-1]
        E = self.energies[i-1]

        x_prev = self.ray_xs[i-2]
        y_prev = self.ray_ys[i-2]
        k_prev = self.ks[i-2]
        l_prev = self.ls[i-2]
        E_prev = self.energies[i-2]

        # ray equation
        # dx/dt = cg
        d = self.bathymetry_interpolate((x, y))
        d_prev = self.bathymetry_interpolate((x_prev, y_prev))
        xnew = x + (self.dt/2) * (3 * self.cg(k, l, d)[0] - self.cg(k_prev, l_prev, d_prev)[0])
        ynew = y + (self.dt/2) * (3 * self.cg(k, l, d)[1] - self.cg(k_prev, l_prev, d_prev)[1])

        # wavenumber equation
        # dk/dt = -grad(sigma)
        sigma_x = (self.dispersion(k, l, self.bathymetry_interpolate((x + self.dx, y))) - self.dispersion(k, l, self.bathymetry_interpolate((x - self.dx, y)))) / (2 * self.dx)
        sigma_y = (self.dispersion(k, l, self.bathymetry_interpolate((x, y + self.dy))) - self.dispersion(k, l, self.bathymetry_interpolate((x, y - self.dy)))) / (2 * self.dy)
        sigma_x_prev = (self.dispersion(k_prev, l_prev, self.bathymetry_interpolate((x_prev + self.dx, y_prev))) - self.dispersion(k_prev, l_prev, self.bathymetry_interpolate((x_prev - self.dx, y_prev)))) / (2 * self.dx)
        sigma_y_prev = (self.dispersion(k_prev, l_prev, self.bathymetry_interpolate((x_prev, y_prev + self.dy))) - self.dispersion(k_prev, l_prev, self.bathymetry_interpolate((x_prev, y_prev - self.dy)))) / (2 * self.dy)
        knew = k - (self.dt/2) * (3 * sigma_x  - sigma_x_prev)
        lnew = l - (self.dt/2) * (3 * sigma_y - sigma_y_prev)

        # energy equation
        # dE/dt = -E div(cg)
        diff_x_cgx = (1 / (2 * self.dx)) * (
                self.cg(k, l, self.bathymetry_interpolate((x + self.dx, y)))[0] - 
                self.cg(k, l, self.bathymetry_interpolate((x - self.dx, y)))[0]
                )
        diff_y_cgy = (1 / (2 * self.dy)) * (
            self.cg(k, l, self.bathymetry_interpolate((x, y + self.dy)))[1] - 
            self.cg(k, l, self.bathymetry_interpolate((x, y - self.dy)))[1]
            )
        diff_x_cgx_prev = (1 / (2 * self.dx)) * (
                self.cg(k_prev, l_prev, self.bathymetry_interpolate((x_prev + self.dx, y_prev)))[0] - 
                self.cg(k_prev, l_prev, self.bathymetry_interpolate((x_prev - self.dx, y_prev)))[0]
                )
        diff_y_cgy_prev = (1 / (2 * self.dy)) * (
            self.cg(k_prev, l_prev, self.bathymetry_interpolate((x_prev, y_prev + self.dy)))[1] - 
            self.cg(k_prev, l_prev, self.bathymetry_interpolate((x_prev, y_prev - self.dy)))[1]
            )
        Enew = E - (self.dt/2) * (
            3 * E * (diff_x_cgx + diff_y_cgy) - 
            E_prev * (diff_x_cgx_prev + diff_y_cgy_prev)
        )
        return xnew, ynew, knew, lnew, Enew
    
    def cg(self, k, l, d):
        '''
        Computes the group velocity
        '''
        dk = 1e-6 * k if k != 0 else 1e-6
        dl = 1e-6 * l if l!= 0 else 1e-6
        sigma_k = (self.dispersion(k + dk, l, d) - self.dispersion(k - dk, l, d)) / (2 * dk)
        sigma_l = (self.dispersion(k, l + dl, d) - self.dispersion(k, l - dl, d)) / (2 * dl)
        return np.array([sigma_k, sigma_l])

    def bounds_check(self, x, y):
        if x < self.x_min or x > self.x_max or y < self.y_min or y > self.y_max:
            warn("The wavefront has moved out of the domain.")
            self.df = pd.DataFrame({
                'time': self.ts,
                'x': self.ray_xs,
                'y': self.ray_ys,
                'k': self.ks,
                'l': self.ls,
                'E': self.energies,
                'A': self.amps
            })
            return True
        elif x - self.dx < self.x_min or x + self.dx > self.x_max or y - self.dy < self.y_min or y + self.dy > self.y_max:
            warn("The wavefront is close to the edge of the domain; numerical errors may occur.")
            self.df = pd.DataFrame({
                'time': self.ts,
                'x': self.ray_xs,
                'y': self.ray_ys,
                'k': self.ks,
                'l': self.ls,
                'E': self.energies,
                'A': self.amps
            })
            return True
        return False



