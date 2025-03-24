'''
This file contains the Particle class which is used to store the properties of a partícles
'''

import numpy as np

class Particle:
    '''
    Class to store the properties of a particle in the simulation.

    Parameters
    ----------
    id : int
        Unique identifier for the particle.
    position : array_like
        Array particle's position.
    properties : dict, optional
        Dictionary of particle's additional properties such as mass, charge...
    
    Attributes
    ----------
    id : int
        Unique identifier for the particle.
    position : ndarray
        Array of particle's position.
    properties : dict
        Dictionary of particle's additional properties such as velocity, mass, charge...

    Methods
    -------
    update_positions(position)
        Update the particle's position.
    '''

    def __init__(self, id: int, position: np.ndarray, properties: dict = None):
        '''
        Initialize a Particle instance.

        Parameters
        ----------
        id : int
            Unique identifier for the particle.
        position : np.ndarray
            Array representing the particle's position.
        velocity : np.ndarray, optional
            Array representing the particle's velocity. Default is None.
        properties : dict, optional
            Dictionary of additional properties for the particle. Default is None.
        '''
        
        self.id = id
        self.position = np.array(position)
        self.properties = properties or {}
        

        

