from particles_mod.core.particle import Particle
import numpy as np

def test_Particle():
    '''
    Test the Particles class to store the positions of particles in the simulation.

    Parameters
    ----------
    None

    Returns
    -------
    None
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

