from particles_mod.core import *
import numpy as np

def test_Particle():
    '''
    
    '''

    # Create a position
    position = np.array([0.5, 0, 0])
    # Create a set of properties
    properties = {'mass': 1.0, 'charge': 1.0}
    # Create a set of particles
    particle = Particle(id =1, position = position, properties = properties)
    # Check the particle's position
    assert np.all(particle.position == position)
    # Check the particle's properties
    assert particle.properties == properties
