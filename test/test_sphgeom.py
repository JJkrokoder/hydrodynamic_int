'''
This script tests the SphereGeometry subclass of the Geometry class.
'''

import numpy as np
from particles_mod.geometry import SphereGeometry

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

    