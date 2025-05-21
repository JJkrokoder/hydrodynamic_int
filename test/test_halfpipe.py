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

    # check if all the first ny particles are at the same x and z coordinates
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
    positions = half_pipe.generate_positions()
    bonds = half_pipe.generate_pairbonds(positions)

    horizontal_distance = half_pipe.HP_length / (half_pipe.ny-1)
    vertical_distance = 2 * half_pipe.HP_radius * np.sin(STheta/(half_pipe.ntheta-1))
    diagonal_distance = np.sqrt(horizontal_distance**2 + vertical_distance**2)

    for i in range(len(bonds)):
        assert np.isclose(bonds[i][3], diagonal_distance) or np.isclose(bonds[i][3], horizontal_distance) or np.isclose(bonds[i][3], vertical_distance), f"Bond {i} distance is incorrect: {bonds[i][3]} != {diagonal_distance} or {horizontal_distance} or {vertical_distance}"

    assert bonds[0][1] == 1, f"Particle 0 is not bonded with particle 1: {bonds[0][1]} != 1"

    for i in range(half_pipe.nparticles):
        assert bonds[i][2] == Kspring, f"Particle {i} spring constant is not correct: {bonds[i][2]} != {Kspring}"
    
    npairbonds = 4*half_pipe.ny*half_pipe.ntheta - 3*half_pipe.ny - 3*half_pipe.ntheta + 2
    assert len(bonds) == npairbonds, f"Number of bonds is incorrect: {len(bonds)} != {half_pipe.nparticles}"
    
    nhorizontalbonds = (half_pipe.ny-1)*half_pipe.ntheta
    nverticalbonds = (half_pipe.ntheta-1)*half_pipe.ny
    ndiagonalbonds = 2*(half_pipe.ntheta-1)*(half_pipe.ny-1)
    horizontalbonds = [bond for bond in bonds if np.isclose(bond[3], horizontal_distance)]
    verticalbonds = [bond for bond in bonds if np.isclose(bond[3], vertical_distance)]
    diagonalbonds = [bond for bond in bonds if np.isclose(bond[3], diagonal_distance)]
    assert len(horizontalbonds) == nhorizontalbonds, f"Number of horizontal bonds is incorrect: {len(horizontalbonds)} != {nhorizontalbonds}"
    assert len(verticalbonds) == nverticalbonds, f"Number of vertical bonds is incorrect: {len(verticalbonds)} != {nverticalbonds}"
    assert len(diagonalbonds) == ndiagonalbonds, f"Number of diagonal bonds is incorrect: {len(diagonalbonds)} != {ndiagonalbonds}"
    
    print("number of particles in y direction: ", half_pipe.ny)

    for i in range(half_pipe.nparticles):
        particle_bonds = [bond for bond in bonds if bond[0] == i or bond[1] == i]
        assert len(particle_bonds) >= 3 and len(particle_bonds) <= 8, f"Particle {i} has an incorrect number of bonds: {len(particle_bonds)} is not between 3 and 8"
        if i == 0 or i == half_pipe.ny-1 or i == half_pipe.nparticles - half_pipe.ny or i == half_pipe.nparticles - 1:
            assert len(particle_bonds) == 3, f"Particle {i} has an incorrect number of bonds: {len(particle_bonds)} != 3"
        elif i % half_pipe.ny == 0 or i % half_pipe.ny == half_pipe.ny - 1 or i // half_pipe.ny == 0 or i // half_pipe.ny == half_pipe.ntheta - 1:
            assert len(particle_bonds) == 5, f"Particle {i} has an incorrect number of bonds: {len(particle_bonds)} != 5"
        else:
            assert len(particle_bonds) == 8, f"Particle {i} has an incorrect number of bonds: {len(particle_bonds)} != 8"
    
def test_HP_angularbonds():
    """
    Test the HalfPipe class for the correct calculation of angular bonds.
    """
    Length = 12.0  
    Radius = 6.0 
    STheta = np.pi / 2
    Kspring = 3.0

    half_pipe = HalfPipe(Length, Radius, STheta, Ka=Kspring)
    positions = half_pipe.generate_positions()
    bonds = half_pipe.generate_anglebonds(positions)

    # check if the first particle is bonded with the second one and the third one
    assert bonds[0][1] == 1, f"Particle 0 is not bonded with particle 1: {bonds[0][1]} != 1"
    assert bonds[0][2] == 2, f"Particle 0 is not bonded with particle 2: {bonds[0][4]} != 2"

    # check if all the bonds spring constant are equal to Kspring
    for i in range(half_pipe.nparticles):
        assert bonds[i][3] == Kspring, f"Particle {i} spring constant is not correct: {bonds[i][2]} != {Kspring}"
    
    nangularbonds = 2*(half_pipe.ny*half_pipe.ntheta - half_pipe.ny - half_pipe.ntheta)
    assert len(bonds) == nangularbonds, f"Number of angular bonds is incorrect: {len(bonds)} != {nangularbonds}"

    vertical_angle = np.pi - 2 * STheta / (half_pipe.ntheta-1)
    nhorizontal_bonds = (half_pipe.ny-2)*half_pipe.ntheta
    nvertical_bonds = (half_pipe.ntheta-2)*(half_pipe.ny)

    horizontal_bonds = [bond for bond in bonds if bond[4] == np.pi]
    vertical_bonds = [bond for bond in bonds if np.isclose(bond[4], vertical_angle)]
    assert len(horizontal_bonds) == nhorizontal_bonds, f"Number of horizontal bonds is incorrect: {len(horizontal_bonds)} != {nhorizontal_bonds}"
    assert len(vertical_bonds) == nvertical_bonds, f"Number of vertical bonds is incorrect: {len(vertical_bonds)} != {nvertical_bonds}"

    for bond in bonds:
        assert np.isclose(bond[4], vertical_angle) or np.isclose(bond[4], np.pi), f"Bond {bond} angle is incorrect: {bond[4]} != {vertical_angle} or {np.pi}"

    assert np.isclose(bonds[0][4], np.pi), f"Angle for bond 0 is incorrect: {bonds[0][4]} != {np.pi/2}"

    two_bonded_idx = [0, half_pipe.ny-1, half_pipe.nparticles - half_pipe.ny, half_pipe.nparticles - 1]
    three_bonded_idx = [1, half_pipe.ny - 2, half_pipe.ny, half_pipe.ny * 2 - 1,
                        half_pipe.nparticles - 2 * half_pipe.ny, half_pipe.nparticles - half_pipe.ny - 1,
                        half_pipe.nparticles - half_pipe.ny + 1, half_pipe.nparticles - 2]
    four_bonded_idx = [half_pipe.ny + 1, half_pipe.ny * 2 - 2,
                       half_pipe.nparticles - 2 * half_pipe.ny + 1, half_pipe.nparticles - half_pipe.ny - 2]
    five_bonded_idx = [half_pipe.ny + 2, half_pipe.ny * 2 - 3,
                       half_pipe.ny * 2 + 1, half_pipe.ny * 3 - 2,
                       half_pipe.nparticles - 3 * half_pipe.ny + 1, half_pipe.nparticles - 2 * half_pipe.ny - 2,
                       half_pipe.nparticles - 2 * half_pipe.ny + 2, half_pipe.nparticles - half_pipe.ny - 3],
    

    for i in range(half_pipe.nparticles):
        particle_bonds = [bond for bond in bonds if bond[0] == i or bond[1] == i or bond[2] == i]
        assert len(particle_bonds) >= 2 and len(particle_bonds) <= 6, f"Particle {i} has an incorrect number of angular bonds: {len(particle_bonds)} is not between 2 and 6"
        if i in two_bonded_idx:
            assert len(particle_bonds) == 2, f"Particle {i} has an incorrect number of angular bonds: {len(particle_bonds)} != 2"
        elif i in three_bonded_idx:
            assert len(particle_bonds) == 3, f"Particle {i} has an incorrect number of angular bonds: {len(particle_bonds)} != 3"
        elif i in four_bonded_idx or i // half_pipe.ny == 0 or i // half_pipe.ny == half_pipe.ntheta - 1 or i % half_pipe.ny == 0 or i % half_pipe.ny == half_pipe.ny - 1:
            assert len(particle_bonds) == 4, f"Particle {i} has an incorrect number of angular bonds: {len(particle_bonds)} != 4"
        elif i in five_bonded_idx or i % half_pipe.ny == 1 or i % half_pipe.ny == half_pipe.ny - 2 or i // half_pipe.ny == 1 or i // half_pipe.ny == half_pipe.ntheta - 2:
            assert len(particle_bonds) == 5, f"Particle {i} has an incorrect number of angular bonds: {len(particle_bonds)} != 5"
        else:
            assert len(particle_bonds) == 6, f"Particle {i} has an incorrect number of angular bonds: {len(particle_bonds)} != 6"

    last_particle_bonds = [bond for bond in bonds if bond[0] == half_pipe.nparticles-1 or bond[1] == half_pipe.nparticles-1 or bond[2] == half_pipe.nparticles-1]
    assert len(last_particle_bonds) == 2, f"Last particle has an incorrect number of angular bonds: {len(last_particle_bonds)} != 2"

def test_HP_strconstr():
    """
    Test the function that constructs the half-pipe positions and bonds.
    """
    Length = 10.0  
    Radius = 5.0 
    STheta = np.pi / 2
    Kspring = 2.0

    positions, bonds = construct_structure(Length, Radius, STheta, Kp=Kspring)

    # check if pairbonds and anglebonds elements of the bond have been created
    assert "pairbonds" in bonds, f"pairbonds not found in bonds dictionary: {bonds}"
    assert "anglebonds" in bonds, f"anglebonds not found in bonds dictionary: {bonds}"

    #check if their types are correct
    assert bonds["pairbonds"]["type"] == ["Bond2", "Harmonic"], f"pairbonds type is incorrect: {bonds['pairbonds']['type']}"
    assert bonds["anglebonds"]["type"] == ["Bond3", "HarmonicAngular"], f"anglebonds type is incorrect: {bonds['anglebonds']['type']}"

    # check if the labels of these elements are correct
    assert bonds["pairbonds"]["labels"] == ["id_i", "id_j", "K", "r0"], f"pairbonds labels are incorrect: {bonds['pairbonds']['labels']}"
    assert bonds["anglebonds"]["labels"] == ["id_i", "id_j", "id_k", "K", "theta0"], f"anglebonds labels are incorrect: {bonds['anglebonds']['labels']}"



