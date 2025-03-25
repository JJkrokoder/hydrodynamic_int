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

    def get_positions(self, particles : Particles):
        '''
        Method to get the positions array of the particles in the sphere.
        
        Parameters:
        -----------
        particles : Particles
            Particles object containing the positions of the particles.

        Returns:
        --------
        numpy.ndarray
            Array containing the positions of the particles in the sphere.
        '''
        # Generate random points in the sphere
        n_points = particles.get_numberparticles()
        random_points = np.random.normal(size=(n_points, 3))
        random_points /= np.linalg.norm(random_points, axis=1)[:, np.newaxis]
        random_points *= self.radius
        random_points += self.center

        return random_points

