import pyUAMMD
import numpy as np
import os
import tempfile
from typing import Iterable

def read_hessian_file(file_path):
    
    hessian_f = np.loadtxt(file_path)
    # Hessian file has shape (npairs, 11), first two columns are the pair indices
    assert (
        hessian_f.shape[1] == 11
    ), f"Hessian file has unexpected shape {hessian_f.shape}"
    # Transform to a (n, n, 3, 3) array
    n = int(np.sqrt(hessian_f.shape[0]))
    assert (
        n * n == hessian_f.shape[0]
    ), f"Unexpected number of pairs {hessian_f.shape[0]}"
    i = hessian_f[:, 0].astype(int)
    j = hessian_f[:, 1].astype(int)
    matrices = hessian_f[:, 2:].reshape(-1, 3, 3)
    hessian = np.empty((n, n, 3, 3))
    hessian[i, j] = matrices
    return hessian

def obtain_Box(positions: Iterable[float]) -> list:
    """
    Obtain the simulation box from the positions of the particles.
    
    Parameters
    ----------
    positions :
        Positions of atoms.
    
    Returns
    -------
    box :
        The simulation box dimensions.
    """

    max_coords = np.max(positions[:][1], axis=0)
    min_coords = np.min(positions[:][1], axis=0)
    box = [(max_coords[i] - min_coords[i])*1.5 for i in range(3)]
    return box

def create_simulation(positions : Iterable[float], bonds: dict, output_file_path: str) -> pyUAMMD.simulation:
    """
    Create a pyUAMMD simulation object with the given positions and bonds. Specifies an
    output file for the Hessian matrix.

    Parameters
    ----------
    positions :
        Positions of atoms. This should be a list of lists or a numpy array of shape (n, 3),
        where n is the number of atoms and 3 represents the x, y, z coordinates.
    bonds :
        A UAMMD-structured dictionary representing the bonds between atoms. This should
        contain information about the types of bonds and their parameters.
    output_file_path :
        The path to the output file for the simulation. This file will be used to
        store the Hessian matrix after the simulation is run.
    
    Returns
    -------
    simulation : pyUAMMD.simulation
        The created pyUAMMD simulation object.

    Example
    -------
    >>> positions = [[0, 0, 0], [1, 1, 1], [2, 2, 2]]
    >>> bonds = {"myBond": {"type": ["Bond", "Harmonic"], "parameters": {}, "labels": ["id_i", "id_j", "K", "r0"]}}
    >>> bonds["myBond"]["data"] = [[0, 1, 1.0, 1.0], [1, 2, 1.0, 1.0]]
    >>> output_file_path = "hessian.txt"
    >>> simulation = create_simulation(positions, bonds, output_file_path)

    Notes
    -----
    All particles are assumed to be of type "A" with a mass of 1.0, radius of 0.5,
    and charge of 0.0.
    """

    # Create a pyUAMMD simulation object
    simulation = pyUAMMD.simulation()

    # Set up the system information
    simulation["system"] = {
        "info": {
            "type": ["Simulation", "Information"],
            "parameters": {"name": "Hessian_Calculation"}
        }
    }

    # Define global parameters
    simulation["global"] = {
        # Set the unit system (in this case, we're using reduced units)
        "units": {"type": ["Units", "None"]},

        # Define particle types
        "types": {
            "type": ["Types", "Basic"],
            "labels": ["name", "mass", "radius", "charge"],
            "data": [["A", 1.0, 0.5, 0.0]]
        },

        # Set the ensemble (NVT: constant Number of particles, Volume, and Temperature)
        "ensemble": {
            "type": ["Ensemble", "NVT"],
            "labels": ["box", "temperature"],
            "data": [[[100, 100, 100], 0.0]]
        }
    }

    # Set up the integrator (Langevin dynamics)
    simulation["integrator"] = {
        "bbk": {
            "type": ["Langevin", "BBK"],
            "parameters": {
                "timeStep": 0.000,
                "frictionConstant": 1.0
            }
        },
        # Define the integration schedule
        "schedule": {
            "type": ["Schedule", "Integrator"],
            "labels": ["order", "integrator", "steps"],
            "data": [[1, "bbk", 1]]
        }
    }

    # Initialize Particle Positions and Topology

    simulation["state"] = {
        "labels": ["id", "position"],
        "data": [[i, positions[i]] for i in range(len(positions))]
    }
    simulation["topology"] = {
        "structure": {
            "labels": ["id", "type"],
            "data": [[i, "A"] for i in range(len(positions))]
        }
    }

    # Initialize the force field dictionary
    simulation["topology"]["forceField"] = bonds
    # Configure Simulation Steps
    simulation["simulationStep"] = {
        # Output simulation information periodically
        "info": {
            "type": ["UtilsStep", "InfoStep"],
            "parameters": {"intervalStep": 1}
        },
        # Output the Hessian matrix
        "hessianMeasure": {
            "type": ["MechanicalMeasure", "HessianMeasure"],
            "parameters": {
                "intervalStep": 1,
                "outputFilePath": output_file_path,
                "mode": "Analytical",
                "outputPrecision": 6
            }
        }
    }
    return simulation


def obtainHessian (positions: Iterable[float] , bonds: dict) -> np.ndarray:
    """
    Obtain the Hessian matrix from the positions and bonds.
    
    Parameters
    ----------
    positions :
        Positions of atoms.
    bonds : dictionary
        A UAMMD-structured dictionary representing the bonds between atoms.
    
    Returns
    -------
    hessian : 
        The Hessian matrix.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        hessian_file_path = os.path.join(tmpdir, "hessian.txt")
        simulation = create_simulation(positions, bonds, hessian_file_path)
        simulation.run()
        hessian = read_hessian_file(hessian_file_path)
        
    return hessian

def diagonalize_hessian(hessian: np.ndarray) -> np.ndarray:
    """
    Diagonalize the Hessian matrix.
    
    Parameters
    ----------
    hessian :
        The Hessian matrix in (nparticles, nparticles, 3, 3) format.
    
    Returns
    -------
    eigenvalues :
        The eigenvalues of the Hessian matrix.
    eigenvectors :
        The eigenvectors of the Hessian matrix in (nparticles * 3, nparticles, 3) format.
    hessian_reshaped :
        The Hessian matrix reshaped to (nparticles * 3, nparticles * 3) format.
    eigenvectors_reshaped :
        The eigenvectors reshaped to (nparticles * 3, nparticles * 3) format.
    """

    nparticles = hessian.shape[0]
    # Transposing is done in order to change al the coordinates and indexes
    # for the second particle before changing the coordinates of the first particle
    preprocessed_hessian = hessian.transpose(0, 2, 1, 3)
    hessian_reshaped = preprocessed_hessian.reshape((nparticles * 3, nparticles * 3)) 
    eigenvalues, eigenvectors = np.linalg.eigh(hessian_reshaped)
    sorted_indices = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors_reshaped = eigenvectors[:, sorted_indices]
    # Reshape eigenvectors set for a mode decomposition
    eigenvectors = eigenvectors_reshaped.T.reshape((nparticles * 3, nparticles, 3))

    return eigenvalues, eigenvectors, hessian_reshaped, eigenvectors_reshaped




