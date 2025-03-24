import numpy as np
from particles_mod.core import Particles

def test_particles():
    # Test the creation of a Particles object
    labels = ['id', 'position']
    data = [0, np.array([0.0, 0.0, 0.0])]
    particles = Particles(labels, data)
    # Check the ids
    assert np.array_equal(particles.ids, np.array([0]))
    # Check the positions
    assert np.array_equal(particles.positions, np.array([[0.0, 0.0, 0.0]]))
    
    # Test the creation of a Particles object with additional properties
    labels = ['id', 'position', 'mass', 'charge']
    data = [0, np.array([0.0, 0.0, 0.0]), 1.0, -1.0]
    particles = Particles(labels, data)
    # Check the additional properties
    assert particles.properties['mass'] == 1.0
    assert particles.properties['charge'] == -1.0

    
    