'''
This file contains the Particle class which is used to store the properties of a system of partícles
'''

import numpy as np
import matplotlib.pyplot as plt
import os

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
    __init__(labels, data)
        Constructor of the Particles class.
    get_numberparticles()
        Method to get the number of particles in the system.
    plot(output_file)
        Method to plot the particles in the system.
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

        # Check if the ids are unique
        if len(np.unique(self.id)) != len(self.id):
            raise ValueError('The ids must be unique.')
    
    def get_numberparticles(self):
        '''
        Method to get the number of particles in the system.
        '''
        return len(self.id)
    
    def set_positions(self, positions : np.ndarray):
        '''
        Method to set the positions of the particles in the system.

        Parameters
        ----------
        positions : numpy.ndarray
            Array containing the new positions of the particles.
        '''
        if positions.shape != self.position.shape:
            raise ValueError('The positions array must have the same shape as the current positions array.')
        
        self.position = positions
    
    def plot(self, output_file : str, remove_file : bool = True):
        '''
        Method to plot the particles in the system.

        Parameters
        ----------
        output_file : str
            String containing the path to the output file.
        '''

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.position[:, 0], self.position[:, 1], self.position[:, 2])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.savefig(output_file)
        plt.close()

        if remove_file:
            os.remove(output_file)
        

    

        
    
                

        
