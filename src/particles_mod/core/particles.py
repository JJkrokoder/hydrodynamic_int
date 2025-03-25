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
        List containing the data lists of the particles (ids and positions are mandatory).
    
    Attributes
    ----------
    labels : list
        List containing the labels of the properties of the particles.
    ids : numpy.ndarray
        Array containing the ids of the particles.
    positions : numpy.ndarray
        Array containing the positions of the particles.
    properties : numpy.ndarray
        Array containing any other optional property of the particles.
    
    Methods
    -------
    '''

    def __init__(self, labels : list, data : list[list]):
        '''
        Constructor of the Particles class.
        '''
        # Check if the labels contain the mandatory labels
        if 'id' not in labels or 'position' not in labels:
            raise ValueError("'id' and 'position' are mandatory labels.")
        
        # Check if the labels and each particle data have the same length for every particle
        data_lengths = np.array([len(d) for d in data])
        if not np.all(data_lengths == len(labels)):
            # The error message shows the id corresponding to the particle with the wrong length
            wrong_length_id = np.where(data_lengths != len(labels))[0][0]
            raise ValueError(f'The labels and data must have the same length. Error for particle with id {data[wrong_length_id][0]}')
        
        # Create the attributes of the class
        self.labels = labels
        for label in labels:
            setattr(self, label, np.array([d[labels.index(label)] for d in data]))


    
                

        
