import hydrodynamic_int.hessian as hess
import numpy as np

def create_particle_pair():
    """
    Create a pair of particles with some position and bond.
    """
    # Create 2 positions
    pos = [[x, [x - 0.5, 0.0, 0.0]] for x in range(2)]
    
    # Create a bonds dictionary
    bonds = {
        "bonds" : {
            "type": ["Bond2", "Harmonic"],
            "parameters": {},
            "labels": ["id_i", "id_j", "K", "r0"],
            "data": [[0, 1, 1.0, 1.0]]
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
    assert np.allclose(hessian, hessian.transpose(1, 0, 3, 2)), "Hessian matrix is not symmetric"


def test_hessian_diagonalization():
    """
    Test the diagonalization of the Hessian matrix.
    """
    hessian = np.array([[[[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[0, 0, 0], [0, 1, 0], [0, 0, 1]]],
                        [[[0, 0, 0], [0, 1, 0], [0, 0, 1]], [[1, 0, 0], [0, 1, 0], [0, 0, 1]]]])

    eigenvalues, eigenvectors, hessian_reshaped, eigenvectors_reshaped = hess.diagonalize_hessian(hessian)

    # check that the first column of the hessian_reshaped is correct
    assert np.allclose(hessian_reshaped[:, 0], hessian[:,0,:,0].flatten()), "First column of hessian_reshaped is not correct"

    # Check that the number of modes is correct
    assert eigenvalues.shape[0] == hessian_reshaped.shape[0], "Number of modes is not correct"
    assert eigenvectors.shape[0] == hessian_reshaped.shape[0], "Number of eigenvectors is not correct"

    # Check that the eigenvalues are real
    assert np.all(np.isreal(eigenvalues)), "Eigenvalues are not real"

    # Check that the eigenvectors matriz is correctly reshaped
    assert eigenvectors_reshaped.shape == (hessian_reshaped.shape[0], hessian_reshaped.shape[0]), "Eigenvectors matrix is not correctly reshaped"
    assert np.allclose(eigenvectors_reshaped, eigenvectors.transpose(1,2,0).reshape(hessian_reshaped.shape[0], hessian_reshaped.shape[0])), "Eigenvectors matrix is not correctly reshaped"

    # Check that the eigenvector matrix is orthogonal
    assert np.allclose(eigenvectors_reshaped @ eigenvectors_reshaped.T, np.eye(eigenvectors_reshaped.shape[0])), "Eigenvectors matrix is not orthogonal"

    # Check that the eigenvectors matrix diagonalize correctly the hessian matrix, resulting in a diagonal matrix with the eigenvalues
    diagonalized_hessian = eigenvectors_reshaped.T @ hessian_reshaped @ eigenvectors_reshaped
    print(diagonalized_hessian)
    print(np.diag(eigenvalues))
    assert np.allclose(diagonalized_hessian, np.diag(eigenvalues)), "Eigenvectors matrix does not diagonalize the hessian matrix correctly"
    
