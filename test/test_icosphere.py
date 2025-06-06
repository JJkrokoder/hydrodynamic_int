import particles_mod.Icosphere as ico
import numpy as np
import tempfile
import os
from typing import Iterable
from scipy.spatial.transform import Rotation as R

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

def test_datageneration():
    '''
    Test the _data_generation method of the IcoSphere class.
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
    
    assert np.all(np.isclose(icosphere.positions, positions_array)), "Positions in IcoSphere and data do not match"
                  


def test_hessian_calculation():
    """
    Test the Hessian calculation for the IcoSphere.
    This test checks the Hessian calculation using both the default method and numerical approximation.
    It also verifies the symmetry of the Hessian and checks that it does not contain translational energy.
    It also checks the modes and eigenvalues obtained from the Hessian.
    """
    
    radius = 2.0
    density = 2.0
    Kpair = 1.0
    Kdi = 1.0

    icosphere = ico.IcoSphere(radius=radius, density=density, Kpair=Kpair, Kdi=Kdi)

    hessian_num = icosphere.calculate_hessian(method='Numerical')

    hessian = icosphere.calculate_hessian()


    assert hessian is not None, "Hessian should not be None after calculation"
    assert hessian_num is not None, "Hessian (numerical) should not be None after calculation"
    assert isinstance(hessian_num, np.ndarray), "Hessian (numerical) should be a numpy array"
    assert isinstance(hessian, np.ndarray), "Hessian should be a numpy array"

    assert hessian.shape == (icosphere.nparticles * 3, icosphere.nparticles * 3), f"Expected Hessian shape {(icosphere.nparticles * 3, icosphere.nparticles * 3)}, got {hessian.shape}"
    assert hessian_num.shape == (icosphere.nparticles * 3, icosphere.nparticles * 3), f"Expected Hessian (numerical) shape {(icosphere.nparticles * 3, icosphere.nparticles * 3)}, got {hessian_num.shape}"

    symmetry_check = np.allclose(hessian, hessian.T, atol=1e-14)
    symmetry_check_num = np.abs(hessian_num - hessian_num.T)
    symmetry_check_num = symmetry_check_num / np.max(np.abs(hessian_num))
    symmetry_check_num = np.allclose(symmetry_check_num, 0, atol=1e-6)
    assert symmetry_check, "Hessian should be symmetric"
    assert symmetry_check_num, "Hessian (numerical) should be symmetric"

    traslational_energy = np.sum(hessian)
    traslational_energy_num = np.sum(hessian_num)
    assert np.isclose(traslational_energy, 0, atol=1e-10), "Hessian should not have traslational energy"
    assert np.isclose(traslational_energy_num, 0, atol=1e-6), "Hessian (numerical) should not have traslational energy"

    modes, eigenvalues = icosphere.obtain_modes()

    assert modes is not None, "Modes should not be None after calculation"
    assert eigenvalues is not None, "Eigenvalues should not be None after calculation"

    assert modes.shape == (icosphere.nparticles * 3, icosphere.nparticles * 3), f"Expected modes shape {(icosphere.nparticles * 3, icosphere.nparticles * 3)}, got {modes.shape}"
    assert eigenvalues.shape == (icosphere.nparticles * 3,), f"Expected eigenvalues shape {(icosphere.nparticles * 3,)}, got {eigenvalues.shape}"

    assert np.all(eigenvalues >= -1e-10), "Eigenvalues should be non-negative"

    orthogonality_check = np.allclose(modes.T @ modes, np.eye(icosphere.nparticles * 3), atol=1e-10)
    assert orthogonality_check, "Modes should be orthogonal"

    diag_hessian = modes.T @ icosphere.hessian @ modes
    assert diag_hessian.shape == (icosphere.nparticles * 3, icosphere.nparticles * 3), f"Expected diagonalized Hessian shape {(icosphere.nparticles * 3, icosphere.nparticles * 3)}, got {diag_hessian.shape}"
    assert np.all(np.isclose(np.diag(diag_hessian), eigenvalues)), "Diagonalized Hessian should match eigenvalues"
    
      
