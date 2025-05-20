from particles_mod.HalfPipe import *
import numpy as np


def test_HPpositions():
    """
    Test the HalfPipe class for the correct calculation of positions.
    """
    Length = 10.0  
    Radius = 5.0 
    STheta = np.pi / 2

    # Create an instance of the HalfPipe class
    half_pipe = HalfPipe(Length, Radius, STheta)
    positions = half_pipe.generate_positions()

    # check the number of particles
    assert len(positions) == half_pipe.nparticles, f"Number of particles is incorrect: {len(positions)} != {half_pipe.nparticles}"

    # check if the y separation between the first two particles is correct
    assert np.isclose(positions[1][1] - positions[0][1], half_pipe.HP_length / (half_pipe.ny-1)), f"y separation is incorrect: {positions[1][1] - positions[0][1]} != {half_pipe.HP_length / half_pipe.ny}"
    # check if all the particles are at a radial distance of from the x=0 and z=Radius axis
    for i in range(half_pipe.nparticles):
        assert np.isclose(np.sqrt(positions[i][0]**2 + (positions[i][2]-Radius)**2), Radius), f"Particle {i} is not at the correct radial distance: {np.sqrt(positions[i][0]**2 + positions[i][2]**2)} != {half_pipe.Radius}"

    # check if all the first ny particles are a the same x and z coordinates
    for i in range(half_pipe.ny):
        assert np.isclose(positions[i][0], -Radius), f"Particle {i} is not at the correct x coordinate: {positions[i][0]} != {-Radius}"
        assert np.isclose(positions[i][2], Radius), f"Particle {i} is not at the correct z coordinate: {positions[i][2]} != {Radius}"

def test_HP_pairbonds():
    """
    Test the HalfPipe class for the correct calculation of pair bonds.
    """
    Length = 10.0  
    Radius = 5.0 
    STheta = np.pi / 2
    Kspring = 2.0

    half_pipe = HalfPipe(Length, Radius, STheta, Kp=Kspring)
    bonds = half_pipe.generate_pairbonds()

    # check if the first particle is bonded with the second one
    assert bonds[0][1] == 1, f"Particle 0 is not bonded with particle 1: {bonds[0][1]} != 1"

    # check if all the bonds spring constant are equal to Kspring
    for i in range(half_pipe.nparticles):
        assert bonds[i][2] == Kspring, f"Particle {i} spring constant is not correct: {bonds[i][2]} != {Kspring}"
    
    # check if the number of bonds is correct
    npairbonds = 4*half_pipe.ny*half_pipe.ntheta - 3*half_pipe.ny - 3*half_pipe.ntheta + 2
    assert len(bonds) == npairbonds, f"Number of bonds is incorrect: {len(bonds)} != {half_pipe.nparticles}"

    # check if the distance for the third bond is correct
    assert np.isclose(bonds[2][3], half_pipe.HP_length / (half_pipe.ny-1)), f"Distance for bond 2 is incorrect: {bonds[2][3]} != {half_pipe.HP_length / (half_pipe.ny-1)}"


