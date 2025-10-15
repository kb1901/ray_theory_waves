import numpy as np


class Vegetation():
    '''
    Create an energy sink term to depict vegetation in a lake
    given a 2d binary array 
    '''
    def __init__(self, 
                 bin_array, 
                 kind
                 ):
        '''
        Parameters
        ----------
        bin_array: binary array depicting 1s for presence of
            vegetation and 0s for absence
        kind: one of "gaussian", ...

        Returns
        -------
        None
        '''
        self.bin_array = bin_array
        self.kind = kind
    
    def __call__(self, x, y):
        '''
        Define function to be called
        '''
        if self.kind =='exponential':
            return self.exponential()
    
    def gaussian(
            self,
            
            ):
        return mean_depth - ridge_height * np.exp(-((xx - mean_x) ** 2 / (2 * sigma_x ** 2) + (yy - mean_y) ** 2 / (2 * sigma_y ** 2)))

