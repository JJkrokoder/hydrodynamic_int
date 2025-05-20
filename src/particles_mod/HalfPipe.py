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
        positions : list
            List of the positions of the half-pipe particles.
        """
        
        self.ny = int(self.HP_length * np.sqrt(self.HP_density))
        self.ntheta = int(np.sqrt(self.HP_density) * self.HP_radius * 2 * self.HP_amplitude)

        self.nparticles = self.ny * self.ntheta
        self.HP_density = self.nparticles / (self.HP_length * self.HP_radius * 2 * self.HP_amplitude)

        # Parametrizer arrays
        y = np.linspace(0, self.HP_length,self.ny)
        theta = np.linspace(-self.HP_amplitude, self.HP_amplitude, self.ntheta)

        return [[self.HP_radius*np.sin(t), j,  self.HP_radius*(1-np.cos(t))] for t in theta for j in y]




def generateHalfPipe(HP_length=1.0, HP_radius=1.0, HP_amplitude=np.pi/2, HP_density=1.0, Kp=1.0, Ka=1.0, gamma=0.0):
    """
    Sets the positions, pair bonds and angular bonds of the half-pipe particles.
    
    Parameters
    ----------
    HP_length : float
        Length of the half-pipe.
    HP_radius : float
        Radius of the half-pipe.
    HP_amplitude : float
        Semiangular amplitude of the half-pipe.
    HP_density : float
        Density of the half-pipe.
    Kp : float
        Spring constant for the pair bonds.
    Ka : float
        Spring constant for the angular bonds.
    gamma : float
        Damping coefficient for the bonds.

    Returns
    -------
    positions : list
        List of the positions of the half-pipe particles.
    pairbonds : list
        List of the pair bonds between the particles.
    anglebonds : list
        List of the angular bonds between the particles.

    Notes
    -----
    The half-pipe is defined by the following parametric equations:
        x(t) = HP_radius * sin(t)
        y(t) = j
        z(t) = HP_radius * (1 - cos(t))
    where t is the angle and j is the length along the half-pipe. This makes the half-pipe to be
    oriented along the y-axis, with the lowest point at (0, 0, 0).
    """
    
    # Calculate the number of particles in the y (length) direction
    ny = int(HP_length * np.sqrt(HP_density))
    # Calculate the number of particles in the theta (angle) direction
    ntheta = int(np.sqrt(HP_density) * HP_radius * 2 * HP_amplitude)

    # Define parametrizer arrays
    y = np.linspace(0, HP_length, ny)   # y-coordinates along the half-pipe axis
    theta = np.linspace(-HP_amplitude, HP_amplitude, ntheta)  # angles in radians

    # obtain the coordinates of the half-pipe
    positions = [[HP_radius*np.sin(t), j,  HP_radius*(1-np.cos(t))] for t in theta for j in y]  # This generates an entire row before changing the angle

    # AVOID PARTICLE OVERLAP IF AMPLITUDE IS CLOSE TO PI MUST BE INCLUDED

    ##############################################################################

    # Create linear pair bonds
    # The bonds are stablished between first neighbours both vertical, horizontal and diagonal in the half-pipe
    
    pairbonds = []

    # Stablish horizontal bonds: ny-1 per row
    for row_id in range(ntheta):
        for column_id in range(ny - 1):
            # Get the current particle id
            particle_id = row_id * ny + column_id
            # obtain the distance between this particle and the next one in the same row
            distance = np.linalg.norm(np.array(positions[particle_id]) - np.array(positions[particle_id + 1]))
            # Add the correpsonding bond information to the list
            pairbonds.append([particle_id, particle_id + 1, Kp, gamma, distance])
            # Assert if the distance is close to HP_length/(ny-1)
            assert np.isclose(distance, HP_length/(ny-1), rtol=0.01), f"Distance between particles {particle_id} and {particle_id + 1} is not close to the expected value."
            

    # Stablish vertical bonds: ntheta-1 per column
    for row_id in range(ntheta - 1):
        for column_id in range(ny):
            # Get the current particle id
            particle_id = row_id * ny + column_id
            # obtain the distance between this particle and the next one in the same column
            distance = np.linalg.norm(np.array(positions[particle_id]) - np.array(positions[particle_id + ny]))
            # Bond the two particles by adding the information to the list
            pairbonds.append([particle_id, particle_id + ny, Kp, gamma, distance])
            # Assert if the distance is close to 2*HP_radius*np.sin(HP_amplitude/(ntheta-1))
            assert np.isclose(distance, 2*HP_radius*np.sin(HP_amplitude/(ntheta-1)), rtol=0.01), f"Distance between particles {particle_id} and {particle_id + ny} is not close to the expected value."
            

    # Stablish diagonal bonds: 2 per cell
    for row_cell_id in range(ntheta - 1):
        for column_cell_id in range(ny - 1):
            # Get the current particle id
            particle_id = row_cell_id * ny + column_cell_id
            # obtain the distance with respect to the diagonal particle
            distance = np.linalg.norm(np.array(positions[particle_id]) - np.array(positions[particle_id + ny + 1]))
            # Bond the two diagonal particles
            pairbonds.append([particle_id, particle_id + ny + 1, Kp, gamma, distance])
            # Bond the two diagonal particles in the other direction
            distance = np.linalg.norm(np.array(positions[particle_id + 1]) - np.array(positions[particle_id + ny]))
            # Bond the two diagonal particles
            pairbonds.append([particle_id + 1, particle_id + ny, Kp, gamma, distance])
    
    ##############################################################################

    # Create angular 3 bonds
    # The bonds are stablished between first neighbours both vertical and horizontal in the half-pipe

    anglebonds = []

    # Create horizontal angular bonds: ny-2 per row
    for row_id in range(ntheta):
        for column_id in range(ny - 2):
            # Get the current particle id
            particle_id = row_id * ny + column_id
            # Get the positions of the three particles
            pos1 = np.array(positions[particle_id])
            pos2 = np.array(positions[particle_id + 1])
            pos3 = np.array(positions[particle_id + 2])
            # Calculate the angle using the dot product
            v1 = pos1 - pos2
            v2 = pos3 - pos2
            angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
            # Bond with the next two particles in the same row
            anglebonds.append([particle_id, particle_id + 1, particle_id + 2, Ka, angle])
            # Assert if the angle is close to pi
            assert np.isclose(angle, np.pi, rtol=0.01), f"Angle between particles {particle_id}, {particle_id + 1} and {particle_id + 2} is not close to pi."

    # Create vertical angular bonds: ntheta-2 per column
    for row_id in range(ntheta - 2):
        for column_id in range(ny):
            # Get the current particle id
            particle_id = row_id * ny + column_id
            # Get the positions of the three particles
            pos1 = np.array(positions[particle_id])
            pos2 = np.array(positions[particle_id + ny])
            pos3 = np.array(positions[particle_id + 2*ny])
            # Calculate the angle using the dot product
            v1 = pos1 - pos2
            v2 = pos3 - pos2
            angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
            # Bond with the next two particles in the same column
            anglebonds.append([particle_id, particle_id + ny, particle_id + 2*ny, Ka, angle])

    return positions, pairbonds, anglebonds



