'''
This file contains the parent definition for the geometry class that manages geometric particle arrangements.
'''

import numpy as np
from particles_mod.core import Particles

class Geometry:
    '''
    Parent class for the geometry classes.    
    
    Parameters:
    -----------
    geometry_name : str
        String containing the name of the geometry.
    parameters : 

    Attributes:
    -----------
    name : str
        String containing the name of the geometry.
    parameters : 
    
    Methods:
    --------
    get_positions(particles : Particles)
        Method to get the positions of the particles in the geometry
    '''
    
    def __init__(self, geometry_name : str, **parameters):
        '''
        Constructor of the Geometry class.
        '''
        self.name = geometry_name
        self.parameters = parameters

    def get_positions(self, particles : Particles):
        '''
        Method to generate a positions array according the geometry characteristics and the attributes of the particles.
        
        Parameters:
        -----------
        particles : Particles
            Particles object containing the positions of the particles.
        
        Returns:
        --------
        numpy.ndarray
            Array containing the positions of the particles in the geometry.
        
        '''
        
        raise NotImplementedError('The get_positions method must be implemented in the child class.')


