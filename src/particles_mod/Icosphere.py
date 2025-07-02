import icosphere as ico
import numpy as np
import VLMP
from typing import Iterable
import tempfile
import json
from hydrodynamic_int.hessian import obtainHessian, diagonalize_hessian, read_hessian_file
import pyUAMMD
from hydrodynamic_int.utils import getMobilityTensor

def create_traslational_mode(nparticles: int, axis: str = "x") -> np.ndarray:
    """
    Create a translational mode for the given number of particles.

    Parameters
    ----------
    nparticles : int
        The number of particles in the system.

    Returns
    -------
    mode_tx : np.ndarray
        A translational mode matrix of shape (nparticles, 3).
    """
    mode_t = np.zeros((nparticles, 3))
    for i in range(nparticles):
        if axis == "x":
            mode_t[i, :] = [1.0, 0.0, 0.0]
        elif axis == "y":
            mode_t[i, :] = [0.0, 1.0, 0.0]
        elif axis == "z":
            mode_t[i, :] = [0.0, 0.0, 1.0]
    
    mode_t = mode_t.flatten()
    
    return mode_t/ np.linalg.norm(mode_t)

def create_rotational_mode(positions: Iterable[float], nparticles: int, axis: str = "x") -> np.ndarray:
    """
    Create a rotational mode for the given number of particles.

    Parameters
    ----------
    positions : Iterable[float]
        The positions of the particles in the system.
    nparticles : int
        The number of particles in the system.
    axis : str, optional
        The axis of rotation, can be "x", "y", or "z". Default is "x".

    Returns
    -------
    mode_r : np.ndarray
        A rotational mode matrix of shape (nparticles, 3).
    """
    mode_r = np.zeros((nparticles, 3))
    for i in range(nparticles):
        if axis == "x":
            mode_r[i, :] = [0.0, -positions[i, 2], positions[i, 1]]
        elif axis == "y":
            mode_r[i, :] = [positions[i, 2], 0.0, -positions[i, 0]]
        elif axis == "z":
            mode_r[i, :] = [-positions[i, 1], positions[i, 0], 0.0]
    mode_r = mode_r.flatten()
    norm = np.linalg.norm(mode_r)
    return mode_r / norm

def create_orthogonal_modes(positions: Iterable[float], modes: np.ndarray) -> np.ndarray:
    """
    Create orthogonal modes from a given set of modes and a translational mode.
    The function uses the Gram-Schmidt process to orthogonalize the modes with respect to the translational mode.
    """
    new_modes = np.copy(modes)
            
    
    new_modes [:,0] = create_traslational_mode(int(modes.shape[0]/3), axis="x")
    new_modes [:,1] = create_traslational_mode(int(modes.shape[0]/3), axis="y")
    new_modes [:,2] = create_traslational_mode(int(modes.shape[0]/3), axis="z")
    new_modes [:,3] = create_rotational_mode(positions, int(modes.shape[0]/3), axis="x")
    new_modes [:,4] = create_rotational_mode(positions, int(modes.shape[0]/3), axis="y")
    new_modes [:,5] = create_rotational_mode(positions, int(modes.shape[0]/3), axis="z")
    

    # Orthogonalize the modes using Gram-Schmidt process
    for i in range(6, modes.shape[1]):
        mode = modes[:, i]
        for j in range(i):
            mode -= np.dot(mode, new_modes[:, j]) * new_modes[:, j]
        mode /= np.linalg.norm(mode)
        new_modes[:, i] = mode
    
    return new_modes


def reconstruct_modes(modes: np.ndarray, mobility_matrix: np.ndarray = None, block_indices: list = [3, 6], method : str = "default", positions: Iterable[float] = None) -> np.ndarray:
    """
    Reconstruct normal modes based on the mobility matrix structure and the provided block indices.
    These block indices indicate groups of modes belonging to a same subspace, which will be diagonalized separately.

    Parameters
    ----------
    modes : 
        The initial modes to be reconstructed.
    mobility_matrix :
        The mobility matrix used to reconstruct the modes.
    block_indices : list, optional
        A list of indices indicating the boundaries of blocks in the mobility matrix.
        Each block corresponds to a set of modes that will be diagonalized separately.

    Returns
    -------
    new_modes :
        The reconstructed modes after applying the mobility matrix and diagonalizing within the specified blocks.
    
        
    Notes
    -----
    The function assumes that the mobility matrix is symmetric and that the modes are orthogonal.
    The default block indices [3, 6] physically correspond to the first six modes, typically translational and rotational modes of the system.
    """

    new_modes = np.copy(modes)

    if method == "default":
        if positions is None:
            raise ValueError("Positions must be provided for default mode reconstruction.")
        new_modes = create_orthogonal_modes(modes=modes, positions=positions)
    else:
        mod_space_mobility = modes.T @ mobility_matrix @ modes
        for block_index in range(len(block_indices)):
            if block_index > 0:
                start, end = block_indices[block_index-1], block_indices[block_index]
            else:
                start, end = 0, block_indices[block_index]
            _, eigenvectors = np.linalg.eigh(mod_space_mobility[start:end, start:end])
            new_modes[:, start:end] = modes[:, start:end] @ eigenvectors
    


    return new_modes


def preliminary_structured_simulation(positions: Iterable[float], bonds: dict):
    """
    Create a preliminary UAMMD-structured simulation dictionary for the icosphere.

    Parameters
    ----------
    positions :
        The positions of the particles in the icosphere.
    bonds : dict
        A UAMMD-structured dictionary representing the bonds between particles.

    Returns
    -------
    simulation :
        A dictionary containing the simulation sections.
    """

    simulation = pyUAMMD.simulation()

    simulation["system"] = {
        "info": {"type": ["Simulation", "Information"], "parameters": {"name": "Icosphere"}}
    }
    
    simulation["global"] = {
        "units": {"type": ["Units", "None"]},
        "types": {"type": ["Types", "Basic"], "labels": ["name", "mass", "radius", "charge"], 
                  "data": [["A", 1.0, 0.5, 0.0]]},
        "ensemble": {"type": ["Ensemble", "NVT"], "labels": ["box", "temperature"], 
                     "data": [[[100, 100, 100], 0.0]]}
    }

    simulation["state"] = {
        "labels": ["id", "position"], 
        "data": [[i, pos] for i, pos in enumerate(positions.tolist())]
    }
    
    simulation["integrator"] = {
        "bbk": {"type": ["Langevin", "BBK"], 
                "parameters": {"timeStep": 0.05, "frictionConstant": 1.0}},
        "schedule": {"type": ["Schedule", "Integrator"], 
                     "labels": ["order", "integrator", "steps"], 
                     "data": [[1, "bbk", 1]]}
    }
    
    simulation["topology"] = {
        "structure": {"labels": ["id", "type"], 
                      "data": [[i, "A"] for i in range(len(positions))]},
        "forceField": bonds
    }

    print("Preliminary simulation structure created")

    return simulation


class IcoSphere:
    '''
    A class to create an icosphere structure with particles.

    Parameters
    ----------
    radius :
        The radius of the icosphere.
    density :
        The surface density of particles in the icosphere.
    Kpair :
        Spring constant for the pair bonds.
    Kdi :
        Spring constant for the dihedral bonds
    '''

    def __init__(self, radius: float = 1.0, density: float = 1.0, Kpair: float = 1.0, Kdi: float = 1.0):
        self.radius = radius
        self.Kpair = Kpair
        self.Kdi = Kdi

        nparticles = round(4 * np.pi * radius**2 * density)
        self.freq_division = round(np.sqrt(1 + (nparticles - 12)/10))

        print(f"Creating icosphere with {nparticles} particles and frequency division {self.freq_division}")
        normalized_positions, self.faces = ico.icosphere(self.freq_division)
        print("Icosphere created with normalized positions and faces.")
        self.positions = normalized_positions * radius
        self.nparticles = len(self.positions)
        print(f"Number of particles: {self.nparticles}")
        self.density = nparticles / (4 * np.pi * radius**2)

        print("Generating VLMP data for the icosphere.")
        vlmp_data = self._generate_data()
        print("VLMP data generated.")
        
        positions = vlmp_data['state']['data']
        positions = [pos[1] for pos in positions]

        bonds = vlmp_data['topology']['forceField']

        self.positions = np.array(positions)
        self.bonds = bonds 

  
    def get_positions(self) -> np.ndarray:
        """
        Get the positions of the particles in the icosphere.

        Returns
        -------
        positions :
            The positions of the particles in the icosphere.
        """
        return self.positions
    
    def set_positions(self, positions: Iterable[float]):
        """
        Set the positions of the particles in the icosphere.

        Parameters
        ----------
        positions :
            The new positions of the particles in the icosphere.
        """
        if len(positions) != self.nparticles:
            raise ValueError(f"Expected {self.nparticles} positions, got {len(positions)}")
        self.positions = positions

    def _generate_data(self) :
        """
        Generate a UAMMD-structured data object for the icosphere.

        Parameters
        ----------
        path :
            The path where the JSON file will be saved.
        """

        temp_file = "current_state"
        simulationPool = [{
            "system": [
                {"type": "simulationName", "parameters": {"simulationName": "Icosphere"}}
            ],
            "units": [{"type": "none"}],
            "types": [{"type": "basic"}],
            "ensemble": [
                {"type": "NVT", "parameters": {"box": [1000.0, 1000.0, 1000.0],
                                            "temperature": 0.0}}
            ],
            "integrators": [
                {"type": "BBK", "parameters": {"timeStep": 0.0,
                                            "frictionConstant": 0.0,
                                            "integrationSteps": 1}}
            ],
            "models": [
                {"type": "ICOSPHERE", "parameters": {
                "resolution": self.freq_division,
                "radius": self.radius,
                "particleName": "S",
                "particleRadius": 0.5,
                "Kb": self.Kpair,
                "Kd": self.Kdi,
                "steric": False
                }},
            ],
            "simulationSteps": [
                {"type": "saveState", "parameters": {"intervalStep": 1,
                                                    "outputFilePath": temp_file,
                                                    "outputFormat": "xyz"}},
            ]
        }]

        # Initialize VLMP and load simulation pool
        vlmp = VLMP.VLMP()
        vlmp.loadSimulationPool(simulationPool)

        # Distribute simulations and set up
        vlmp.distributeSimulationPool("size", 1)

        with tempfile.TemporaryDirectory() as dir:
            vlmp.setUpSimulation(dir)

            json_path = f"{dir}/simulationSets/simulationSet_0/Icosphere/simulation.json"

            with open(json_path, 'r') as json_file:
                data = json.load(json_file)

        return data

    
    
    def calculate_hessian(self, method: str = 'Analytical', equilibrium : bool = True) -> np.ndarray:
        '''
        Calculate the Hessian matrix for the icosphere structure. If the system relaxation
        is ensured before calculating the Hessian, must be updated.

        Parameters
        ----------
        method :
            The method to use for Hessian calculation. Options are 'Analytical' or 'Numerical'.

        Returns
        -------
        hessian :
            The Hessian matrix of the icosphere structure.
        '''

        simulation = preliminary_structured_simulation(positions=self.positions, bonds=self.bonds)

        relaxation_steps = 0

        if equilibrium:
            relaxation_steps = self.nparticles * 5
        
        with tempfile.TemporaryDirectory() as output_dir:
        
            simulation['simulationStep']= {
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
            
            if equilibrium:
                simulation['integrator']['schedule']['data'][0][2] = relaxation_steps + 1
                
                simulation['simulationStep']['positionsmeasure'] = {
                    "type": ["WriteStep", "WriteStep"],
                    "parameters": {
                        "intervalStep": relaxation_steps,
                        "startStep": 1,
                        "outputFilePath": f"{output_dir}/positions",
                        "outputFormat": "sp",
                        "pbc": True
                    }
                }

                simulation['simulationStep']['forcemeasure'] = {
                    "type": ["MechanicalMeasure", "PairwiseForceMeasure"],
                    "parameters": {
                        "intervalStep": relaxation_steps,
                        "startStep": 1,
                        "outputFilePath": f"{output_dir}/forces.txt",
                        "mode": "Total_force",
                    }
                }
            
            
            
            simulation.run()
            hessian = read_hessian_file(f"{output_dir}/hessian.txt")
            self.positions = np.loadtxt(f"{output_dir}/positions.sp", skiprows=1, usecols=(0, 1, 2))
            self.forces = np.loadtxt(f"{output_dir}/forces.txt", skiprows=1, usecols=(1, 2, 3))
        
        
        return hessian.transpose(0, 2, 1, 3).reshape(self.nparticles * 3, self.nparticles * 3)
     
    def obtain_modes(self, method: str = 'Analytical') -> np.ndarray:
        """
        Obtain the modes of the icosphere structure.

        Parameters
        ----------
        method :
            The method to use for Hessian calculation. Options are 'Analytical' or 'Numerical'.

        Returns
        -------
        modes :
            The modes of the icosphere structure.
        eigenvalues :
            The eigenvalues of the Hessian matrix.
        
        """
        if hasattr(self, 'hessian'):
            if self.hessian.shape != (self.nparticles * 3, self.nparticles * 3):
                hessian = self.hessian.transpose(0, 2, 1, 3).reshape(self.nparticles * 3, self.nparticles * 3)
        else:
            hessian = self.calculate_hessian(method=method)
            self.hessian = hessian
        eigenvalues, modes = diagonalize_hessian(hessian)

        self.modes = modes
        self.eigenvalues = eigenvalues

    def obtain_normal_coupled_mobility(self, solver, method: str = 'Analytical', reconstr_modes: bool = False) -> np.ndarray:
        """
        Obtain the normal coupled mobility matrix of the icosphere structure.

        Returns
        -------
        mobility_matrix :
            The normal coupled mobility matrix of the icosphere structure.

        """

        if not hasattr(self, 'modes'):
            self.obtain_modes(method=method)

        if reconstr_modes:
            if not hasattr(self, 'mobility_matrix'):
                self.mobility_matrix = getMobilityTensor(self.positions, solver=solver)
            self.modes = reconstruct_modes(modes=self.modes, mobility_matrix=self.mobility_matrix, method = "mobility")
        mobility_matrix = getMobilityTensor(self.positions, solver=solver)
        
        modes = np.copy(self.modes)
        coupled_mobility = modes.T @ mobility_matrix @ modes
        return coupled_mobility
    


    





