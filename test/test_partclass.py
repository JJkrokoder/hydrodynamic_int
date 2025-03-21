from particles_mod.core import *
import numpy as np

def test_Particle():
    '''
    
    '''

    # Create a position
    position = np.array([0.5, 0, 0])
    # Create a velocity
    velocity = np.array([1, 1, 1])
    # Create a set of properties
    properties = {'mass': 1.0, 'charge': 1.0}
    # Create a set of particles
    particle = Particle(position = position, velocity = velocity, properties = properties)
    # Check the particle's position
    assert np.all(particle.position == position)
    # Check the particle's velocity
    assert np.all(particle.velocity == velocity)
    # Check the particle's properties
    assert particle.properties == properties
    
    # Update the particle's position
    timestep = 0.5
    particle.update_position(timestep)
    # Check the particle's new position
    assert np.all(particle.position == position + velocity * timestep)

    # Test the particles constructor
    particle1 = Particle(position = np.array([0.5, 0, 0]), velocity = np.array([1, 1, 1]))
    particle2 = Particle(position = np.array([0.5, 0.5, 0]), velocity = np.array([1, 1, 1]))
    particles = Particles([particle1, particle2])
    # Check positions array shape
    assert particles.positions.shape == (2, particle1.position.shape[0])
    # Check the particles positions
    assert np.all(particles.positions == np.array([[0.5, 0, 0], [0.5, 0.5, 0]]))
    # Check the particles velocities
    assert np.all(particles.velocities == np.array([[1, 1, 1], [1, 1, 1]]))
