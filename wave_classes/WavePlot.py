from wave_classes.CartWaveSol import CartWaveSol
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import seaborn as sns
import numpy as np
import pandas as pd
sns.set_style('white')

class WavePlot:

    def __init__(
            self,
            fig,
            ax,
            cart_wave_sol: CartWaveSol | list[CartWaveSol]
    ):
        if type(cart_wave_sol) != list:
            self.cart_wave_sol = [cart_wave_sol]
        else:    
            self.cart_wave_sol = cart_wave_sol
        self.ax = ax
        self.fig = fig
    
    def plot_wave_traj(
            self, 
            xx, 
            yy, 
            bathymetry, 
            color,
            arrow: bool=True,
            label=None, 
            save_to=None
            ):
        '''
        Plots the wave trajectories in the object passed
        '''
        if not self.ax.has_data():
            im = self.ax.pcolormesh(xx, yy, bathymetry, cmap='Reds')
            self.fig.colorbar(im, label='Bathmetry (m)')
        max_amps = []
        for sol in self.cart_wave_sol:
            max_amps.append(np.max(sol.df['A']))
        max_amp = np.max(max_amps)
        for sol in self.cart_wave_sol:
            df = sol.df
            s1 = df['A'].to_numpy()/max_amp
            if arrow:
                self.arrow(df['x'], df['y'], self.ax, 5)
            self.ax.scatter(df['x'], df['y'], c=color, label=label, edgecolors='black', s=self.map_range(s1, 10, 80))
            self.ax.legend(loc='upper right')
            self.ax.xaxis.set_tick_params(direction='in')
            self.ax.yaxis.set_tick_params(direction='in')
        if save_to:
            plt.savefig(save_to)
        return None
    
    def plot_bath(self, xx, yy, bathymetry):
        if not self.ax.has_data():
            im = self.ax.pcolormesh(xx, yy, bathymetry, cmap='Reds')
            self.fig.colorbar(im, label='Bathmetry (m)')
        return None
    
    # helpers
    @staticmethod
    def map_range(src, new_min, new_max):
        src_min = np.min(src)
        src_max = np.max(src)
        output = (
            ((src - src_min) / (src_max - src_min)) * 
            (new_max - new_min) + new_min
        )
        return output

    @staticmethod
    def arrow(x,y,ax,n):
        d = len(x)//(n+1)    
        ind = np.arange(d,len(x),d)
        for i in ind:
            ar = FancyArrowPatch((x[i-1],y[i-1]),(x[i],y[i]), 
                                arrowstyle='->', mutation_scale=50)
            ax.add_patch(ar)