'''
Sphere Geometry Subclass
'''

import numpy as np
from particles_mod.geometry.general import Geometry
from particles_mod.core import Particles

class SphereGeometry(Geometry):
    '''
    Class for the sphere geometry.
    
    Parameters:
    -----------
    radius : float
        Radius of the sphere.
    center : array_like
        Center of the sphere.

    Attributes:
    -----------
    radius : float
        Radius of the sphere.
    center : array_like
        Center of the sphere.

    Methods:
    --------
    set_positions(particles : Particles)
        Method to set the positions of the particles in the sphere.
    '''

    def __init__(self, radius : float, center : np.ndarray):
        '''
        Constructor of the SphereGeometry class.
        '''
        super().__init__('sphere')
        self.radius = radius
        self.center = center
        


