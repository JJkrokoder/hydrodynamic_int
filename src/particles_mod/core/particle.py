'''
This file contains the Particle class which is used to store the properties of a partícles
'''

import numpy as np

class Particle:
    '''
    Class to store the properties of a particle in the simulation.

    Parameters
    ----------
    position : array_like
        Array particle's position.
    properties : dict, optional
        Dictionary of particle's additional properties such as mass, charge...
    
    Attributes
    ----------
    positions : ndarray
        Array of particle's position.
    properties : dict
        Dictionary of particle's additional properties such as mass, charge...

    Methods
    -------
    update_positions(position)
        Update the particle's position.
    '''

    def __init__(self, position: np.ndarray, velocity: np.ndarray = None, properties: dict = None):
        
        # Check the velocity array shape
        if velocity is not None:
            if len(position) != len(velocity): raise ValueError('Position and velocity must have the same length.')

        self.position = np.array(position)
        self.velocity = np.array(velocity) if velocity is not None else None
        self.properties = properties or {}
        
    def update_position(self, timestep: float):
        '''
        Update the particle's position based on its own velocity.

        Parameters
        ----------
        time_step : float
            Time step of the simulation.


        Examples
        --------
        >>> particle = Particle(position = np.array([0.5, 0, 0]), velocity = np.array([1, 1, 1]))
        >>> particle.update_position(0.5)
        >>> particle.position
        array([0.5, 0, 0])
        '''
        
        if self.velocity is not None:
            self.position += self.velocity * timestep
        else:
            raise ValueError('Particle velocity is not defined.')

        

