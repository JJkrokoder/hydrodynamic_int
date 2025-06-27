import particles_mod.Icosphere as ico
import numpy as np
import tempfile
import os
from typing import Iterable
from scipy.spatial.transform import Rotation as R
from pytest import mark
import libMobility as lm


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
    
    radius = 3.0
    density = 1.0
    Kpair = 1.0
    Kdi = 1.0

    icosphere = ico.IcoSphere(radius=radius, density=density, Kpair=Kpair, Kdi=Kdi)

    hessian_num = icosphere.calculate_hessian(method='Numerical')
    
    hessian = icosphere.calculate_hessian()

    assert icosphere.positions.shape == (icosphere.nparticles, 3), f"Expected positions shape {(icosphere.nparticles, 3)}, got {icosphere.positions.shape}"
    
    assert icosphere.forces is not None, "Forces should not be None after initialization"
    assert isinstance(icosphere.forces, np.ndarray), "Forces should be a numpy array"
    assert icosphere.forces.shape == (icosphere.nparticles, 3), f"Expected forces shape {(icosphere.nparticles, 3)}, got {icosphere.forces.shape}"
    assert np.all(np.isclose(icosphere.forces, 0, atol=1e-9)), "Forces should be zero after initialization"

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


def test_modes_and_eigenvalues():
    """
    Test the calculation of modes and eigenvalues from the Hessian of the IcoSphere.
    This test checks that the modes and eigenvalues are calculated correctly,
    that the modes are orthogonal, and that the diagonalized Hessian matches the eigenvalues.
    """
    radius = 3.0
    density = 1.0
    Kpair = 1.0
    Kdi = 1.0
    icosphere = ico.IcoSphere(radius=radius, density=density, Kpair=Kpair, Kdi=Kdi)

    icosphere.obtain_modes()

    modes = icosphere.modes
    eigenvalues = icosphere.eigenvalues

    assert modes is not None, "Modes should not be None after calculation"
    assert eigenvalues is not None, "Eigenvalues should not be None after calculation"

    assert modes.shape == (icosphere.nparticles * 3, icosphere.nparticles * 3), f"Expected modes shape {(icosphere.nparticles * 3, icosphere.nparticles * 3)}, got {modes.shape}"
    assert eigenvalues.shape == (icosphere.nparticles * 3,), f"Expected eigenvalues shape {(icosphere.nparticles * 3,)}, got {eigenvalues.shape}"

    assert np.all(np.diff(eigenvalues) >= 0), "Eigenvalues should be sorted in non-decreasing order"
    assert np.all(eigenvalues >= -1e-8), "Eigenvalues should be non-negative"

    orthogonality_check = np.allclose(modes.T @ modes, np.eye(icosphere.nparticles * 3), atol=1e-10)
    assert orthogonality_check, "Modes should be orthogonal"

    diag_hessian = modes.T @ icosphere.hessian @ modes
    assert diag_hessian.shape == (icosphere.nparticles * 3, icosphere.nparticles * 3), f"Expected diagonalized Hessian shape {(icosphere.nparticles * 3, icosphere.nparticles * 3)}, got {diag_hessian.shape}"
    assert np.all(np.isclose(np.diag(diag_hessian), eigenvalues)), "Diagonalized Hessian should match eigenvalues"


def generate_orthogonal_matrix(size, seed=None):
    """
    Generate a random orthogonal matrix of the specified size.
    """
    if seed is not None:
        np.random.seed(seed)
    random_matrix = np.random.rand(size, size)
    q, r = np.linalg.qr(random_matrix)
    return q

def generate_symmetric_matrix(size, seed=None):
    """
    Generate a random symmetric matrix of the specified size.
    """
    if seed is not None:
        np.random.seed(seed)
    random_matrix = np.random.rand(size, size)
    symmetric_matrix = (random_matrix + random_matrix.T) / 2
    return symmetric_matrix


@mark.parametrize("matrix_size", [2, 3, 5])
def test_reconstruct_modes_without_internal_diag(matrix_size):
    """
    Test the reconstruction of modes without internal diagonalization.
    """
    modes = generate_orthogonal_matrix(matrix_size)
    mobility = generate_symmetric_matrix(matrix_size)

    new_modes = ico.reconstruct_modes(modes = modes, mobility_matrix = mobility, block_indices=[])

    assert np.all(modes == new_modes), "Reconstructed modes should match original modes"


@mark.parametrize("matrix_size", [4, 6, 7])
@mark.parametrize("block_indices", [[], [2, 3]])
def test_reconstructed_modes_orthogonality(matrix_size, block_indices):
    """
    Test the orthogonality of reconstructed modes with specified block indices.
    """
    modes = generate_orthogonal_matrix(matrix_size)
    mobility = generate_symmetric_matrix(matrix_size)

    new_modes = ico.reconstruct_modes(modes = modes, mobility_matrix = mobility, block_indices = block_indices)

    assert np.allclose(new_modes.T @ new_modes, np.eye(matrix_size), atol=1e-10), \
        "Reconstructed modes should be orthogonal"


@mark.parametrize("matrix_size", [6, 7, 8])
@mark.parametrize("block_indices", [[3], [2, 4, 5]])
def test_new_modes_block_deco(matrix_size, block_indices):
    """
    Test that the new modes obtained from reconstruct_modes have the correct diagonal block decomposition
    """
    modes = generate_orthogonal_matrix(matrix_size)
    mobility = generate_symmetric_matrix(matrix_size)

    new_modes = ico.reconstruct_modes(modes = modes, mobility_matrix = mobility, block_indices = block_indices)
    new_basis_mobility = new_modes.T @ mobility @ new_modes

    for block_index in range(len(block_indices)):
        end = block_indices[block_index]
        if block_index == 0:
            start = 0
        else:
            start = block_indices[block_index - 1]
        
        block = new_basis_mobility[start:end, start:end]
        diagonal_block = np.diag(np.diag(block))
        assert np.allclose(block, diagonal_block), \
            f"Block {block_index} of mobility should be diagonal in the new modes basis"


def create_default_solver(method="SelfMobility", wallheight: float = 0.0):
    """Create a default solver for testing purposes."""
    if method == "SelfMobility":
        solver = lm.SelfMobility("open", "open", "open")
        solver.setParameters(5)
    elif method == "NBody":
        solver = lm.NBody("open", "open", "open")
        solver.setParameters()
    elif method == "NBodywall":
        solver = lm.NBody("open", "open", "single_wall")
        solver.setParameters(wallHeight=wallheight)
    solver.initialize(
        temperature=0,
        viscosity=1/(6 * np.pi),  # Viscosity for a sphere in a fluid
        hydrodynamicRadius=1.0,  # Default hydrodynamic radius
        needsTorque=False,
    )
    return solver

@mark.parametrize("radius, density, Kpair, Kdi", [
    (5.0, 1.0, 0.5, 1.0),
    (10.0, 0.2, 1.0, 1.5),
    (3.0, 0.5, 0.2, 0.8)
])
@mark.parametrize("method", ["NBody", "SelfMobility", "NBodywall"])
@mark.parametrize("wallheight", [20, 30, 50])
def test_coupled_mobility_symmetry(radius, density, Kpair, Kdi, method, wallheight):
    """
    Test the symmetry of the coupled mobility matrix.
    """
    
    icosphere = ico.IcoSphere(radius=radius, density=density, Kpair=Kpair, Kdi=Kdi)
    solver = create_default_solver(method=method, wallheight=-wallheight)
    mobility_matrix = icosphere.obtain_normal_coupled_mobility(solver=solver)

    assert np.allclose(mobility_matrix, mobility_matrix.T, atol=1e-7, rtol=1e-7), "Mobility matrix should be symmetric"
