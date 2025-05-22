import numpy as np
from typing import Iterable


class Plain_Membrane:
    """
    Class to generate a Plane Membrane structure with particles.
    
    Parameters
    ----------
    LengthX :
        Length of the membrane in the X direction.
    LengthY :
        Length of the membrane in the Y direction.
    Density :
        Particle density of the membrane.
    Kp :
        Spring constant for the pair bonds.
    Ka :
        Spring constant for the angular bonds.
    """

    def __init__(self, LengthX: float = 5.0, LengthY: float = 5.0, Density: float = 1.0, Kp: float = 1.0, Ka: float = 1.0):
        self.LengthX = LengthX
        self.LengthY = LengthY
        self.Density = Density
        self.Kp = Kp
        self.Ka = Ka

        # Calculate the number of particles based on the density
        self.nx = int(np.sqrt(self.Density) * self.LengthX)
        self.ny = int(np.sqrt(self.Density) * self.LengthY)
        self.nparticles = self.nx * self.ny
        self.dx = self.LengthX / (self.nx - 1)
        self.dy = self.LengthY / (self.ny - 1)
        self.density = 1/(self.dx * self.dy)
    
    def generate_positions(self) -> Iterable[float] :
        """
        Generates the positions of the particles in the half-pipe structure.
        
        Returns
        -------
        positions :
            Positions of the membrane particles.
        """
        
        # Parametrizer arrays
        y = [-self.LengthY/2 + i * self.dy for i in range(self.ny)]
        x = [-self.LengthX/2 + i * self.dx for i in range(self.nx)]
        positions = [[i, j, 0] for i in x for j in y]
        return positions

    def generate_pairbonds(self, positions: Iterable[float]) -> Iterable[float]:
        """
        Generates the pair and angular bonds between the particles in the membrane structure. 
        This seems to be general for any open squared parametrized structure, such as a half-pipe or a strip.
        The function generates pair bonds between adjacent particles in the x and y directions, as well as diagonal bonds.

        Parameters
        ----------
        positions :
            Positions of the membrane particles.
        
        Returns
        -------
        pairbonds :
            Pair bonds between the particles.
        """
        
        pairbonds = []

        # Horizontal bonds
        for row_id in range(self.nx):
            for column_id in range(self.ny - 1):
                particle_id = row_id * self.ny + column_id
                distance = np.linalg.norm(np.array(positions[particle_id]) - np.array(positions[particle_id + 1]))
                pairbonds.append([particle_id, particle_id + 1, self.Kp, distance])

        # Vertical bonds
        for row_id in range(self.nx - 1):
            for column_id in range(self.ny):
                particle_id = row_id * self.ny + column_id
                distance = np.linalg.norm(np.array(positions[particle_id]) - np.array(positions[particle_id + self.ny]))
                pairbonds.append([particle_id, particle_id + self.ny, self.Kp, distance])

        # Diagonal bonds
        for row_cell_id in range(self.nx - 1):
            for column_cell_id in range(self.ny - 1):
                particle_id = row_cell_id * self.ny + column_cell_id
                distance = np.linalg.norm(np.array(positions[particle_id]) - np.array(positions[particle_id + self.ny + 1]))
                pairbonds.append([particle_id, particle_id + self.ny + 1, self.Kp, distance])
                distance = np.linalg.norm(np.array(positions[particle_id + 1]) - np.array(positions[particle_id + self.ny]))
                pairbonds.append([particle_id + 1, particle_id + self.ny, self.Kp, distance])

        return pairbonds
    
    def generate_anglebonds(self, positions: list) -> list:
        """
        Generates the angular bonds between the particles in the membrane structure.

        Parameters
        ----------
        positions :
            Positions of the membrane particles.
        
        Returns
        -------
        anglebonds :
            Angular bonds between the particles.
        """
        
        anglebonds = []

        # Horizontal angular bonds
        for row_id in range(self.nx):
            for column_id in range(self.ny - 2):
                particle_id = row_id * self.ny + column_id
                pos1 = np.array(positions[particle_id])
                pos2 = np.array(positions[particle_id + 1])
                pos3 = np.array(positions[particle_id + 2])
                v1 = pos1 - pos2
                v2 = pos3 - pos2
                angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
                anglebonds.append([particle_id, particle_id + 1, particle_id + 2, self.Ka, angle])

        # Vertical angular bonds
        for row_id in range(self.nx - 2):
            for column_id in range(self.ny):
                particle_id = row_id * self.ny + column_id
                pos1 = np.array(positions[particle_id])
                pos2 = np.array(positions[particle_id + self.ny])
                pos3 = np.array(positions[particle_id + 2*self.ny])
                v1 = pos1 - pos2
                v2 = pos3 - pos2
                angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
                anglebonds.append([particle_id, particle_id + self.ny, particle_id + 2*self.ny, self.Ka, angle])

        return anglebonds

def construct_structure(LengthX: float, LengthY: float, Density: float, Kp: float, Ka: float) -> tuple:
    """
    Constructs the structure of the membrane and generates the positions and bonds.

    Parameters
    ----------
    LengthX :
        Length of the membrane in the X direction.
    LengthY :
        Length of the membrane in the Y direction.
    Density :
        Particle density of the membrane.
    Kp :
        Spring constant for the pair bonds.
    Ka :
        Spring constant for the angular bonds.

    Returns
    -------
    positions :
        Positions of the membrane particles.
    bonds :
        Dictionary containing the pair and angular bonds.
    """
    
    membrane = Plain_Membrane(LengthX=LengthX, LengthY=LengthY, Density=Density, Kp=Kp, Ka=Ka)
    positions = membrane.generate_positions()
    pairbonds = membrane.generate_pairbonds(positions)
    anglebonds = membrane.generate_anglebonds(positions)

    # Create a bonds dictionary
    bonds = {
        "pairbonds" : {
            "type": ["Bond2", "Harmonic"],
            "parameters": {},
            "labels": ["id_i", "id_j", "K", "r0"],
            "data": pairbonds
        },
        "anglebonds" : {
            "type": ["Bond3", "HarmonicAngular"],
            "parameters": {},
            "labels": ["id_i", "id_j", "id_k", "K", "theta0"],
            "data": anglebonds
        }
    }

    return positions, bonds