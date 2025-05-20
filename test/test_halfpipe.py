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

def test_HP_pairbonds()