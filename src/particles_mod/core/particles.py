'''
This file contains the Particle class which is used to store the properties of a system of partícles
'''

import numpy as np

class Particles:
    '''
    Class to store the properties of a system of particles in the simulation.

    Parameters
    ----------
    labels : list
        List containing the labels of the properties of the particles (ids and positions are mandatory).
    data: list
        List containing the data of the particles (ids and positions are mandatory).
    
    Attributes
    ----------
    ids : numpy.ndarray
        Array containing the ids of the particles.
    positions : numpy.ndarray
        Array containing the positions of the particles.
    properties : dict
        Dictionary containing optional properties of the particles.
    
    Methods
    -------
    '''

    def __init__(self, labels : list, data : np.ndarray):
        '''
        Constructor of the Particles class.
        '''
        # Check if the labels and data have the same length
        if len(labels) != len(data):
            raise ValueError('The labels and data must have the same length.')
        # Check if the labels contain the mandatory labels
        if 'id' not in labels:
            raise ValueError('The labels must contain the id label.')
        if 'position' not in labels:
            raise ValueError('The labels must contain the position label.')
        
        # Check that the first element is an integer id and the second element is a position array
        if not isinstance(data[0], int):
            raise ValueError('The first element of the data must be an integer id.')
        if not isinstance(data[1], np.ndarray):
            raise ValueError('The second element of the data must be a position array.')
        
        # Initialize the ids, positions and properties
        self.ids = np.array([data[0]])
        self.positions = np.array([data[1]])
        self.properties = {}
        # Add the rest of the data to the properties
        for i in range(2, len(labels)):
            self.properties[labels[i]] = data[i]
        
        


        
