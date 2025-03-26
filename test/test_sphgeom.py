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
    ########################################################
