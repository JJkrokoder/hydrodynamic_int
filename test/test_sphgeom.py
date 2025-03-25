'''
This script tests the SphereGeometry subclass of the Geometry class.
'''

import numpy as np
from particles_mod.geometry import SphereGeometry
from particles_mod.core import Particles

def test_sphere_geometry():
    '''
    This function tests the SphereGeometry subclass of the Geometry class.
    '''
    # Test SphereGeometry initialization
    radius = 1.0
    center = np.array([0.0, 0.0, 0.0])
    sphere = SphereGeometry(radius, center)
    assert sphere.name == 'sphere'
    assert sphere.radius == radius
    assert np.all(sphere.center == center)
    ########################################################

    # Test SphereGeometry get_positions
    # Create a Particles object
    labels = ['id', 'position']
    data = [[0, np.array([0.0, 0.0, 0.0])],
            [1, np.array([1.0, 1.0, 1.0])],
            [2, np.array([2.0, 2.0, 2.0])]]
    particles = Particles(labels, data)
    # Create a SphereGeometry object
    radius = 1.0
    center = np.array([0.0, 0.0, 0.0])
    sphere = SphereGeometry(radius, center)
    # Get the positions
    positions = sphere.get_positions(particles)
    # Check the positions
    assert positions.shape == (3, 3)
    assert np.all(np.linalg.norm(positions - center, axis=1) <= radius)

    # Create and plot a particles object with 100 particles in a sphere of radius 2 centered at [1.0, 0.0, 0.0]
    labels = ['id', 'position']
    number_particles = 500
    data = [[i, np.array([0.0, 0.0, 0.0])] for i in range(number_particles)]
    particles = Particles(labels, data)
    center = np.array([1.0, 0.0, 0.0])
    radius = 2.0
    sphere = SphereGeometry(radius, center)
    positions = sphere.get_positions(particles)
    particles.set_positions(positions)
    particles.plot('test/output/sphere.png', remove_file=False)
    ########################################################
