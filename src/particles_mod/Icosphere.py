import icosphere as ico
import numpy as np
from typing import Iterable

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
    Kangle :
        Spring constant for the angular bonds.
    '''

    def __init__(self, radius: float = 1.0, density: float = 1.0, Kpair: float = 1.0, Kangle: float = 1.0):
        self.radius = radius
        self.Kpair = Kpair
        self.Kangle = Kangle

        nparticles = round(4 * np.pi * radius**2 * density)
        self.nu = round(np.sqrt(1 + (nparticles - 12)/10))

        normalized_positions, self.faces = ico.icosphere(self.nu)
        self.positions = normalized_positions * radius
        self.nparticles = len(self.positions)
        self.density = nparticles / (4 * np.pi * radius**2)

    def generate_positions(self) -> Iterable[float]:
        """
        Generates the positions of the particles in the icosphere structure.

        Returns
        -------
        positions :
            Positions of the icosphere particles.
        """
        return self.positions.tolist()





