import pyUAMMD
import numpy as np
import os
import tempfile
from typing import Iterable
#import cupy as cp

def read_hessian_file(file_path):
    
    hessian_f = np.loadtxt(file_path)
    print(f"Read Hessian file {file_path} with shape {hessian_f.shape}")
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


def create_simulation(positions : Iterable[float], bonds: dict, output_dir: str, method : str = "Numerical") -> pyUAMMD.simulation:
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

    relaxation_steps = 1000

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
            "data": [[[1000, 1000, 1000], 0.0]]
        }
    }

    # Set up the integrator (Langevin dynamics)
    simulation["integrator"] = {
        "bbk": {
            "type": ["Langevin", "BBK"],
            "parameters": {
                "timeStep": 0.05,
                "frictionConstant": 1.0
            }
        },
        # Define the integration schedule
        "schedule": {
            "type": ["Schedule", "Integrator"],
            "labels": ["order", "integrator", "steps"],
            "data": [[1, "bbk", relaxation_steps + 1]]
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
        # Output the Hessian matrix
        "hessianmeasure": {
            "type": ["MechanicalMeasure", "HessianMeasure"],
            "parameters": {
                "intervalStep": relaxation_steps,
                "outputFilePath": f"{output_dir}/hessian.txt",
                "mode": method,
                "outputPrecision": 15,
                "startStep": 1
            }
        }
    }

    simulation["simulationStep"]["positions"] = {
        "type": ["WriteStep", "WriteStep"],
        "parameters": {
            "intervalStep": relaxation_steps,
            "startStep": 1,
            "outputFilePath": f"{output_dir}/positions",
            "outputFormat": "sp",
            "pbc": True
        }
    }

    simulation["simulationStep"]["ForceMeasure"] = {
        "type": ["MechanicalMeasure", "PairwiseForceMeasure"],
        "parameters": {
            "intervalStep": relaxation_steps,
            "startStep": 1,
            "outputFilePath": f"{output_dir}/forces.txt",
            "mode": "Total_force",
        }
    }

    return simulation


def obtainHessian (positions: Iterable[float] , bonds: dict, create_sp : bool = False, method : str = "Numerical") -> np.ndarray:
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
        if create_sp:
            current_dir = os.getcwd()
            simulation = create_simulation(positions, bonds, current_dir, method = method)
        else:
            simulation = create_simulation(positions, bonds, tmpdir, method = method)
        simulation.run()
        hessian = read_hessian_file(f"{tmpdir}/hessian.txt")
        
    return hessian

def diagonalize_hessian(hessian: np.ndarray) -> np.ndarray:
    """
    Diagonalize the Hessian matrix.
    
    Parameters
    ----------
    hessian :
        The Hessian matrix in (nparticles * 3, nparticles * 3) format.
    
    Returns
    -------
    eigenvalues :
        The eigenvalues of the Hessian matrix.
    eigenvectors :
        The eigenvectors of the Hessian matrix in (nparticles * 3, nparticles * 3) format.
    """

    
    hessian = (hessian + hessian.T) / 2  # Ensure symmetry
    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    sorted_indices = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    return eigenvalues, eigenvectors




