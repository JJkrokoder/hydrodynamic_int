'''
This file contains the Particles class which is used to store the positions of particles in the simulation.
'''

import numpy as np

class Particles:
    '''
    Class to store the positions of particles in the simulation.

    Parameters
    ----------
    positions : array_like
        Array of positions of the particles.
    properties : dict, optional
        Dictionary of properties of the particles.
    
    Attributes
    ----------
    positions : ndarray
        Array of positions of the particles.
    properties : dict
        Dictionary of properties of the particles.

    Methods
    -------
    add_particles(positions)
        Add particles to the current list of particles.
    remove_particles(indices)
        Remove particles from the current list of particles.

    Examples
    --------
    >>> particles = Particles([[0, 0], [1, 1]])
    >>> particles.add_particles([[2, 2], [3, 3]])
    >>> particles.positions
    array([[0, 0],
           [1, 1],
           [2, 2],
           [3, 3]])
    >>> particles.remove_particles([0, 1])
    >>> particles.positions
    array([[2, 2],
           [3, 3]])
    '''

    def __init__(self, positions, properties=None):
        self.positions = np.array(positions)
        self.properties = properties or {}
        
    def add_particles(self, positions):
        self.positions = np.vstack([self.positions, positions])
        
    def remove_particles(self, indices):
        self.positions = np.delete(self.positions, indices, axis=0)

