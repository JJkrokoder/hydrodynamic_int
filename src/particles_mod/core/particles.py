'''
This file contains the Particle class which is used to store the properties of a system of partícles
'''

import numpy as np
from particles_mod.core.particle import Particle

class Particles:
    '''
    Class to store the properties of a system of particles in the simulation.

    Parameters
    ----------
    particles : list
        List of Particle objects.

    Attributes
    ----------
    particles : list
        List of Particle objects.
    positions : ndarray
        Array of particle's positions.
    velocities : ndarray
        Array of particle's velocities.
    properties : dict
        Dictionary of particle's additional properties such as mass, charge...

    Methods
    '''

    def __init__(self, particles: list[Particle]):
        self.positions = np.array([particle.position for particle in particles]).reshape(len(particles), len(particles[0].position))
        self.velocities = np.array([particle.velocity for particle in particles]).reshape(len(particles), len(particles[0].velocity))
