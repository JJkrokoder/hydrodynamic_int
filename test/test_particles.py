import numpy as np
from particles_mod.core import Particles

def test_particles():
    
    # Check if the error is raised when any of the particles data have different lengths than the labels
    labels = ['id', 'position']
    data = [[3, np.array([0.0, 0.0, 0.0]), 1.0], [1, np.array([1.0, 1.0, 1.0])]]
    try:
        particles = Particles(labels, data)
    except ValueError as e:
        assert str(e) == 'The labels and data must have the same length. Error for particle with id 3'
    ####################################################################################################

    # Check if the error is raised when the labels do not contain the mandatory labels
    labels = ['position']
    data = [0]
    try:
        particles = Particles(labels, data)
    except ValueError as e:
        assert str(e) == "'id' and 'position' are mandatory labels."
    ####################################################################################################

    # Test the creation of a Particles object
    labels = ['id', 'position']
    data = [[0, np.array([0.0, 0.0, 0.0])],
            [1, np.array([1.0, 1.0, 1.0])],
            [2, np.array([2.0, 2.0, 2.0])]]
    particles = Particles(labels, data)
    # Check the ids
    assert np.all(particles.id == np.array([0, 1, 2]))
    # Check the positions
    assert np.all(particles.position == np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]))
    ####################################################################################################

    # Test the creation of a Particles object with additional properties
    labels = ['id', 'position', 'property', 'material']
    data = [[0, np.array([0.0, 0.0, 0.0]), 1.0, 'A'],
            [1, np.array([1.0, 1.0, 1.0]), 2.0, 'B'],
            [2, np.array([2.0, 2.0, 2.0]), 3.0, 'C']]
    particles = Particles(labels, data)
    # Check the ids
    assert np.all(particles.id == np.array([0, 1, 2]))
    # Check the positions
    assert np.all(particles.position == np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]))
    # Check the properties
    assert np.all(particles.property == np.array([1.0, 2.0, 3.0]))
    # Check the materials
    assert np.all(particles.material == np.array(['A', 'B', 'C']))
    ####################################################################################################

    # Test the creation of a Particles object with additional properties and different order
    labels = ['id', 'material', 'position', 'property']
    data = [[0, 'A', np.array([0.0, 0.0, 0.0]), 1.0],
            [1, 'B', np.array([1.0, 1.0, 1.0]), 2.0],
            [2, 'C', np.array([2.0, 2.0, 2.0]), 3.0]]
    particles = Particles(labels, data)
    # Check the ids
    assert np.all(particles.id == np.array([0, 1, 2]))
    # Check the positions
    assert np.all(particles.position == np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]))
    # Check the properties
    assert np.all(particles.property == np.array([1.0, 2.0, 3.0]))
    # Check the materials
    assert np.all(particles.material == np.array(['A', 'B', 'C']))
    ####################################################################################################
    
    # Test the uniqueness of the ids
    labels = ['id', 'position']
    data = [[0, np.array([0.0, 0.0, 0.0])],
            [0, np.array([1.0, 1.0, 1.0])]]
    try:
        particles = Particles(labels, data)
    except ValueError as e:
        assert str(e) == 'The ids must be unique.'
    ####################################################################################################

    # Test the get_numberparticles method
    labels = ['id', 'position']
    data = [[0, np.array([0.0, 0.0, 0.0])],
            [1, np.array([1.0, 1.0, 1.0])],
            [2, np.array([2.0, 2.0, 2.0])]]
    particles = Particles(labels, data)
    assert particles.get_numberparticles() == 3
    ####################################################################################################

    # Test the plot method
    labels = ['id', 'position']
    data = [[0, np.array([0.0, 0.0, 0.0])],
            [1, np.array([1.0, 1.0, 1.0])],
            [2, np.array([2.0, 2.0, 2.0])]]
    particles = Particles(labels, data)
    particles.plot('test/output/particles.png')
    ####################################################################################################

    # Test the set_positions method
    labels = ['id', 'position']
    data = [[0, np.array([0.0, 0.0, 0.0])],
            [1, np.array([1.0, 1.0, 1.0])],
            [2, np.array([2.0, 2.0, 2.0])]]
    particles = Particles(labels, data)
    new_positions = np.array([[1.0, 0.0, 0.0], [2.0, 1.0, 1.0], [3.0, 2.0, 2.0]])
    particles.set_positions(new_positions)
    assert np.all(particles.position == new_positions)
    ####################################################################################################


    
    