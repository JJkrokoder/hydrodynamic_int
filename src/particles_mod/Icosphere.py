import icosphere as ico
import numpy as np
import VLMP
from typing import Iterable
import tempfile
import json


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

        normalized_positions, self.faces = ico.icosphere(self.freq_division)
        self.positions = normalized_positions * radius
        self.nparticles = len(self.positions)
        self.density = nparticles / (4 * np.pi * radius**2)
    

    
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
                {"type": "NVT", "parameters": {"box": [100.0, 100.0, 100.0],
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
                {"type": "saveState", "parameters": {"intervalStep": 0,
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

def construct_structure(radius: float = 1.0, density: float = 1.0, Kpair: float = 1.0, Kdi: float = 1.0) -> tuple:
    """
    Constructs the structure of the icosphere and generates the positions and bonds.

    Parameters
    ----------
    radius :
        The radius of the icosphere.
    density :
        The surface density of particles in the icosphere.
    Kpair :
        Spring constant for the pair bonds.
    Kdi :
        Spring constant for the dihedral bonds.

    Returns
    -------
    positions :
        Positions of the icosphere particles.
    bonds :
        Dictionary containing the pair bonds.
    """
    
    icosphere = IcoSphere(radius=radius, density=density, Kpair=Kpair, Kdi=Kdi)
    vlmp_data = icosphere._generate_data()
    
    positions = vlmp_data['state']['data']
    positions = [pos[1] for pos in positions]

    bonds = vlmp_data['topology']['forceField']

    print(bonds.keys())

    return positions, bonds





