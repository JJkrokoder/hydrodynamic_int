from particles_mod.core.particles import Particles
import numpy as np

def test_Particles():
    '''
    Test the Particles class to store the positions of particles in the simulation.

    Parameters
    ----------
    None

    Returns
    -------
    None
    '''

    # Create a set of positions
    positions = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    # Create a set of properties
    properties = {'mass': 1.0, 'charge': 1.0}
    # Create a set of particles
    particles = Particles(positions, properties)
    # Add particles to the current list of particles
    particles.add_particles([[4, 4], [5, 5]])
    # Remove particles from the current list of particles
    particles.remove_particles([0, 1])

    # Check the positions of the particles
    assert np.allclose(particles.positions, np.array([[2, 2], [3, 3], [4, 4], [5, 5]]))

    # Check the properties of the particles
    assert particles.properties == properties

