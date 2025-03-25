'''
This is a file for testing the general geometry class.
'''

import numpy as np
from particles_mod.geometry.general import Geometry
from particles_mod.core import Particles

def test_geometry():
    '''
    Test the general geometry class.
    '''

    # Create a Geometry object
    geometry = Geometry('test', {'param1': 1})
    #geometry.set_positions(particles)
    
    # Test the attributes
