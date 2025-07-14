import hydrodynamic_int.hessian as hess
import numpy as np

def create_particle_pair():
    """
    Create a pair of particles with some position and bond.
    """
    # Create 2 positions
    pos = [[x - 1.0, 0.0, 0.0] for x in range(3)]
    
    # Create a bonds dictionary
    bonds = {
        "bonds" : {
            "type": ["Bond2", "Harmonic"],
            "parameters": {},
            "labels": ["id_i", "id_j", "K", "r0"],
            "data": [[0, 1, 1.0, 1.0],
                     [1, 2, 1.0, 1.0]]
        }
    }
    
    return pos, bonds


def test_hessian_symmetry():
    """
    Test the symmetry of the Hessian matrix.
    """
    # Create a position and bonds lists

    positions, bonds = create_particle_pair()

    # Obtain the Hessian matrix
    hessian = hess.obtainHessian(positions, bonds)

    # Check if the Hessian has been created
    assert hessian is not None, "Hessian matrix is None"

    # check the shape of the Hessian matrix
    nparticles = len(positions)
    assert hessian.shape == (nparticles, nparticles, 3, 3), f"Hessian matrix has incorrect shape: {hessian.shape}"

    # Check hessian symmetry
    assert np.allclose(hessian, hessian.transpose(1, 0, 3, 2), atol=1e-6, rtol=1e-6), "Hessian matrix is not symmetric"


def test_hessian_diagonalization():
    """
    Test the diagonalization of the Hessian matrix.
    """
    hessian = np.array([[[[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[0, 0, 0], [0, 1, 0], [0, 0, 1]]],
                        [[[0, 0, 0], [0, 1, 0], [0, 0, 1]], [[1, 0, 0], [0, 1, 0], [0, 0, 1]]]])
    
    nparticles = hessian.shape[0]
    number_of_modes = nparticles * 3
    
    hessian = hessian.transpose(0, 2, 1, 3).reshape(6, 6)

    eigenvalues, eigenvectors = hess.diagonalize_hessian(hessian)

    assert hessian.shape == (number_of_modes, number_of_modes), "Hessian reshaped has incorrect shape"
    assert eigenvectors.shape == (number_of_modes, number_of_modes), "Eigenvectors reshaped has incorrect shape"
    assert eigenvalues.shape == (number_of_modes,), "Eigenvalues has incorrect shape"
   
    
    assert np.all(np.isreal(eigenvalues)), "Eigenvalues are not real"

    orthogonality = eigenvectors.T @ eigenvectors - np.eye(number_of_modes)
    assert np.allclose(orthogonality, 0), "Eigenvectors reshaped are not orthogonal"
    
    diagonalized_hessian = eigenvectors.T @ hessian @ eigenvectors
    assert np.allclose(diagonalized_hessian, np.diag(eigenvalues)), "Eigenvectors matrix does not diagonalize the hessian matrix correctly"
    




