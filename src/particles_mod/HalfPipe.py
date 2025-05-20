import numpy as np


class HalfPipe:
    """
    Class to generate a half-pipe structure with particles.
    
    Parameters
    ----------
    HP_length :
        Length of the half-pipe.
    HP_radius :
        Radius of the half-pipe.
    HP_amplitude :
        Semiangular amplitude of the half-pipe.
    HP_density :
        Density of the half-pipe.
    Kp :
        Spring constant for the pair bonds.
    Ka :
        Spring constant for the angular bonds.
    """

    def __init__(self, HP_length: float = 1.0, HP_radius: float = 1.0, HP_amplitude: float = np.pi/2, HP_density: float = 1.0, Kp: float = 1.0, Ka: float = 1.0):
        """
        Initializes the HalfPipe class with the given parameters.
        """
        self.HP_length = HP_length
        self.HP_radius = HP_radius
        self.HP_amplitude = HP_amplitude
        self.HP_density = HP_density
        self.Kp = Kp
        self.Ka = Ka
    
    def generate_positions(self):
        """
        Generates the positions of the particles in the half-pipe structure.
        
        Returns
        -------
        positions :
            Positions of the half-pipe particles.
        """
        
        self.ny = int(self.HP_length * np.sqrt(self.HP_density))
        self.ntheta = int(np.sqrt(self.HP_density) * self.HP_radius * 2 * self.HP_amplitude)

        self.nparticles = self.ny * self.ntheta
        self.HP_density = self.nparticles / (self.HP_length * self.HP_radius * 2 * self.HP_amplitude)

        # Parametrizer arrays
        y = np.linspace(0, self.HP_length,self.ny)
        theta = np.linspace(-self.HP_amplitude, self.HP_amplitude, self.ntheta)

        return [[self.HP_radius*np.sin(t), j,  self.HP_radius*(1-np.cos(t))] for t in theta for j in y]
    
    def generate_pairbonds(self, positions: list) -> list:
        """
        Generates the pair and angular bonds between the particles in the half-pipe structure.

        Parameters
        ----------
        positions :
            Positions of the half-pipe particles.
        
        Returns
        -------
        pairbonds :
            Pair bonds between the particles.
        """
        
        positions = self.generate_positions()
        
        pairbonds = []

        # Horizontal bonds
        for row_id in range(self.ntheta):
            for column_id in range(self.ny - 1):
                particle_id = row_id * self.ny + column_id
                distance = np.linalg.norm(np.array(positions[particle_id]) - np.array(positions[particle_id + 1]))
                pairbonds.append([particle_id, particle_id + 1, self.Kp, distance])

        # Vertical bonds
        for row_id in range(self.ntheta - 1):
            for column_id in range(self.ny):
                particle_id = row_id * self.ny + column_id
                distance = np.linalg.norm(np.array(positions[particle_id]) - np.array(positions[particle_id + self.ny]))
                pairbonds.append([particle_id, particle_id + self.ny, self.Kp, distance])

        # Diagonal bonds
        for row_cell_id in range(self.ntheta - 1):
            for column_cell_id in range(self.ny - 1):
                particle_id = row_cell_id * self.ny + column_cell_id
                distance = np.linalg.norm(np.array(positions[particle_id]) - np.array(positions[particle_id + self.ny + 1]))
                pairbonds.append([particle_id, particle_id + self.ny + 1, self.Kp, distance])
                distance = np.linalg.norm(np.array(positions[particle_id + 1]) - np.array(positions[particle_id + self.ny]))
                pairbonds.append([particle_id + 1, particle_id + self.ny, self.Kp, distance])

        return pairbonds
    
    def generate_anglebonds(self, positions: list) -> list:
        """
        Generates the angular bonds between the particles in the half-pipe structure.

        Parameters
        ----------
        positions :
            Positions of the half-pipe particles.
        
        Returns
        -------
        anglebonds :
            Angular bonds between the particles.
        """
        
        anglebonds = []

        # Horizontal angular bonds
        for row_id in range(self.ntheta):
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
        for row_id in range(self.ntheta - 2):
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

def construct_structure(length: float = 1.0, radius: float = 1.0, amplitude: float = np.pi/2, density: float = 1.0, Kp: float = 1.0, Ka: float = 1.0) -> tuple:
    """
    Constructs a half-pipe structure with the given parameters.

    Parameters
    ----------
    length :
        Length of the half-pipe.
    radius :
        Radius of the half-pipe.
    amplitude :
        Semiangular amplitude of the half-pipe.
    density :
        Density of the half-pipe.
    Kp : float, optional
        Spring constant for the pair bonds (default is 1.0).
    Ka : float, optional
        Spring constant for the angular bonds (default is 1.0).

    Returns
    -------
    tuple :
        A tuple containing the positions and bonds of the half-pipe structure.
    """
    
    half_pipe = HalfPipe(length, radius, amplitude, density, Kp, Ka)
    positions = half_pipe.generate_positions()
    pairbonds = half_pipe.generate_pairbonds(positions)
    anglebonds = half_pipe.generate_anglebonds(positions)

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