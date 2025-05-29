import particles_mod.Icosphere as ico
import numpy as np
import tempfile
import os

def test_icosphere():
    
    r = 2.0
    d = 3.0
    Kpair = 0.5

    nparticles = round(4 * np.pi * r**2 * d)
    
    icosphere = ico.IcoSphere(radius=r, density=d, Kpair=Kpair)

    assert icosphere.radius == r, f"Expected radius {r}, got {icosphere.radius}"
    assert icosphere.Kpair == Kpair, f"Expected Kpair {Kpair}, got {icosphere.Kpair}"
    assert icosphere.Kdi == 1.0, f"Expected Kdi 1.0, got {icosphere.Kdi}"

    assert np.isclose(icosphere.nparticles, nparticles, rtol=1e-1), f"Expected nparticles {nparticles}, got {icosphere.nparticles}"
    assert np.isclose(icosphere.density, d, rtol=1e-2), f"Expected density {d}, got {icosphere.density}"

    distances = np.linalg.norm(icosphere.positions, axis=1)
    assert np.all(np.isclose(distances, r, atol=1e-12)), "Not all particles are at the expected radius"

def test_JSONgeneration():
    '''
    Test the JSON generation of the IcoSphere class
    '''

    r = 2.0
    d = 3.0
    Kpair = 1.5
    icosphere = ico.IcoSphere(radius=r, density=d, Kpair=Kpair)

    data = icosphere._generate_data()

    assert isinstance(data, dict), "Data should be a dictionary"
    assert 'topology' in data, "Data should contain 'topology' key"
    assert 'state' in data, "Data should contain 'state' key"
    assert 'system' in data, "Data should contain 'system' key"
    assert 'global' in data, "Data should contain 'global' key"
    assert 'integrator' in data, "Data should contain 'integrator' key"
    assert 'simulationStep' in data, "Data should contain 'simulationStep' key"
    
    nparticles_obtained = len(data['state']['data'])
    assert nparticles_obtained == icosphere.nparticles, f"Expected {icosphere.nparticles} particles, got {nparticles_obtained}"
    for i in range(icosphere.nparticles):
        assert len(data['state']['data'][i]) == 2, f"Particle {i} data should have 2 elements, got {len(data['state']['data'][i])}"
        assert len(data['state']['data'][i][1]) == 3, f"Particle {i} position should have 3 coordinates, got {len(data['state']['data'][i][1])}"

    positions_array = np.array([data['state']['data'][i][1] for i in range(icosphere.nparticles)])
    assert np.all(np.isclose(np.linalg.norm(positions_array, axis=1), r, atol=1e-12)), "Not all particles are at the expected radius in JSON data"

    assert positions_array.shape == (icosphere.nparticles, 3), f"Expected positions array shape {(icosphere.nparticles, 3)}, got {positions_array.shape}"
    
    assert np.all(np.isclose(icosphere.positions, positions_array)), "Positions in JSON data do not match IcoSphere positions"
                  

def test_construct_structure():
    """
    Test the construct_structure function with a known configuration.
    """
    radius = 2.0
    density = 3.0
    Kpair = 1.0
    Kdi = 1.0

    nparticles = round(4 * np.pi * radius**2 * density)
    freq_division = round(np.sqrt(1 + (nparticles - 12)/10))
    nparticles = 12 + 10 *(freq_division**2 - 1)

    positions, bonds = ico.construct_structure(radius = radius, density = density, Kpair=Kpair, Kdi=Kdi)

    assert len(positions) == nparticles, f"Expected {nparticles} particles, got {len(positions)}"
    assert isinstance(bonds, dict), "Bonds should be a dictionary"


    
    
    
      
