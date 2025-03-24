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
    parameters : dict
        Dictionary containing the parameters of the geometry.

    Attributes:
    -----------
    name : str
        String containing the name of the geometry.
    parameters : dict
        Dictionary containing the parameters of the geometry.
    
    Methods:
    --------
    set_positions(particles : Particles)
        Method to set the positions of the particles in the geometry
    '''
    
    def __init__(self, geometry_name : str, parameters : dict):
        '''
        Constructor of the Geometry class.
        '''
        self.name = geometry_name
        self.parameters = parameters
        self.particles = None

    def set_positions(self, particles : Particles):
        '''
        Method to set the positions of the particles in the geometry.
        
        Parameters:
        -----------
        particles : Particles
            Particles object containing the properties of the particles.
        
        '''
        self.particles = particles


