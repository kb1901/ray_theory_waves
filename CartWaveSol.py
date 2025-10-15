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
                 vegetation,
                 x0, 
                 y0, 
                 E0, 
                 T, 
                 dt,
                 dispersion=None
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
        # derived quantities
        self.ts = np.arange(0, T, dt)
        self.dx = x[1] - x[0]
        self.dy = y[1] - y[0]


        # interpolate
        self.bathymetry_interpolate = RGI((x, y), self.bathymetry.T, method='linear')
 
        # initial amplitude
        d_init = self.bathymetry_interpolate((self.x0, self.y0))
        self.a0 = np.sqrt(2 * self.E0) / self.dispersion(self.init_k[0], self.init_k[1], d_init)

        
        # self.x_interpolate = RGI((x, y), self.x.T)
        # self.y_interpolate = RGI((x, y), self.y.T)

        

        # empty arrays for later
        # wave vector change over time
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


            # adding to dataframe
            cur_row = {
                'time': [self.ts[i-1]],
                'x': [self.ray_xs[i-1]],
                'y': [self.ray_ys[i-1]],
                'k': [self.ks[i-1]],
                'l': [self.ls[i-1]],
                'E': [self.energies[i-1]],
                'A': [self.amps[i-1]]
                }
            df_row = pd.DataFrame(cur_row)
            self.df = pd.concat([self.df, df_row], axis=0, ignore_index=True)

            k = self.ks[i-1]
            l = self.ls[i-1]
            x = self.ray_xs[i-1]
            y = self.ray_ys[i-1]
            E = self.energies[i-1]

            dk = 1e-6 * k if k != 0 else 1e-6
            dl = 1e-6 * l if l != 0 else 1e-6


            # calculate depth based on previous coordinates
            try:
                d = self.bathymetry_interpolate((x, y))
            except:
                warn("The wavefront may be out of the domain.")
                return None
            
            # evolve the ray position
            xnow = x + self.dt * self.cg(k, l, d)[0]
            ynow = y + self.dt * self.cg(k, l, d)[1]

            self.ray_xs[i] = xnow
            self.ray_ys[i] = ynow

            # if we want to use updated positions to update wave vector
            if symp:
                x = xnow
                y = ynow

            # calculate the horizontal derivative of the dispersion relation
            try:
                sigma_x = (self.dispersion(k, l, self.bathymetry_interpolate((x + self.dx, y))) - self.dispersion(k, l, self.bathymetry_interpolate((x - self.dx, y)))) / (2 * self.dx)
                sigma_y = (self.dispersion(k, l, self.bathymetry_interpolate((x, y + self.dy))) - self.dispersion(k, l, self.bathymetry_interpolate((x, y - self.dy)))) / (2 * self.dy)
            
            except:
                warn("The wavefront may be out of the domain.")
                return None

            # evolve the wave numbers
            know = k - sigma_x * self.dt
            lnow = l - sigma_y * self.dt
            self.ks[i] = know
            self.ls[i] = lnow

            # if we want to use updated wave vector and position to update the energy
            if symp:
                k = know
                l = lnow
            # Recalculate the bathymetry for the energy/amplitude update
            try:
                dnow = self.bathymetry_interpolate((x, y))
            except:
                warn("The wavefront may be out of the domain.")
                return None
            
            diff_x_cgx = (1 / (2 * self.dx)) * (
                self.cg(k, l, self.bathymetry_interpolate((x + self.dx, y)))[0] - 
                self.cg(k, l, self.bathymetry_interpolate((x - self.dx, y)))[0]
                )
            diff_y_cgy = (1 / (2 * self.dy)) * (
                self.cg(k, l, self.bathymetry_interpolate((x, y + self.dy)))[1] - 
                self.cg(k, l, self.bathymetry_interpolate((x, y - self.dy)))[1]
                )
            E_now = E - self.dt * E * (diff_x_cgx + diff_y_cgy)
            self.energies[i] = E_now

            amp_now = np.sqrt(2 * E_now) / (self.dispersion(know, lnow, dnow))
            self.amps[i] = amp_now
            
        return None
    
    def cg(self, k, l, d):
        '''
        Computes the group velocity
        '''
        # const = np.sqrt((self.g * d) / (k **2 + l ** 2)) 
        # k_mag = np.sqrt(k**2 + l**2)
        # sigma = dispersion(k, l, d)
        # cg_k = 1/((2 * sigma)) * (self.g* k * (np.tanh(k_mag * d)) / k_mag + self.g * (1/np.cosh(k_mag * d)) ** 2 * k * d)
        # cg_l = 1/((2 * sigma)) * (self.g* l * (np.tanh(k_mag * d)) / k_mag + self.g * (1/np.cosh(k_mag * d)) ** 2 * l * d) 
        dk = 1e-6 * k if k != 0 else 1e-6
        dl = 1e-6 * l if l!= 0 else 1e-6
        sigma_k = (self.dispersion(k + dk, l, d) - self.dispersion(k - dk, l, d)) / (2 * dk)
        sigma_l = (self.dispersion(k, l + dl, d) - self.dispersion(k, l - dl, d)) / (2 * dl)
        return np.array([sigma_k, sigma_l])
        # return np.array([cg_k, cg_l])



