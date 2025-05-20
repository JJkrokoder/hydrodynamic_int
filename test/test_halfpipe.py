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
    positions = half_pipe.generate_positions()
    bonds = half_pipe.generate_pairbonds(positions)

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

    # Search for the bonds where particle npaticles-ny appears
    selected_bonds = [bond for bond in bonds if (bond[0] == half_pipe.nparticles - half_pipe.ny or bond[1] == half_pipe.nparticles - half_pipe.ny)]
    assert len(selected_bonds) == 3, f"Particle {half_pipe.nparticles - half_pipe.ny} should be bonded with 3 particles, but found {len(selected_bonds)} bonds: {selected_bonds}"

    selected_bonds = [bond for bond in bonds if (bond[0] == half_pipe.nparticles - 1 or bond[1] == half_pipe.nparticles - 1)]
    assert len(selected_bonds) == 3, f"Particle {half_pipe.nparticles - 1} should be bonded with 3 particles, but found {len(selected_bonds)} bonds: {selected_bonds}"
    # check if this particles is bonded with nparticles - 2, nparticles - half_pipe.ny - 2 and nparticles - half_pipe.ny - 1
    assert (half_pipe.nparticles - 2) in [bond[0] for bond in selected_bonds] or (half_pipe.nparticles - 2) in [bond[1] for bond in selected_bonds], f"Particle {half_pipe.nparticles - 2} should be bonded with particle {half_pipe.nparticles - 1}, but not found in the bonds: {selected_bonds}"
    assert (half_pipe.nparticles - half_pipe.ny - 2) in [bond[0] for bond in selected_bonds] or (half_pipe.nparticles - half_pipe.ny - 2) in [bond[1] for bond in selected_bonds], f"Particle {half_pipe.nparticles - half_pipe.ny - 2} should be bonded with particle {half_pipe.nparticles - 1}, but not found in the bonds: {selected_bonds}"
    assert (half_pipe.nparticles - half_pipe.ny - 1) in [bond[0] for bond in selected_bonds] or (half_pipe.nparticles - half_pipe.ny - 1) in [bond[1] for bond in selected_bonds], f"Particle {half_pipe.nparticles - half_pipe.ny - 1} should be bonded with particle {half_pipe.nparticles - 1}, but not found in the bonds: {selected_bonds}"


def test_HP_angularbonds():
    """
    Test the HalfPipe class for the correct calculation of angular bonds.
    """
    Length = 10.0  
    Radius = 5.0 
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
    
    # check if the number of bonds is correct
    nangularbonds = 2*(half_pipe.ny*half_pipe.ntheta - half_pipe.ny - half_pipe.ntheta)
    assert len(bonds) == nangularbonds, f"Number of angular bonds is incorrect: {len(bonds)} != {nangularbonds}"

    # check if the angle for the first bond is correct
    assert np.isclose(bonds[0][4], np.pi), f"Angle for bond 0 is incorrect: {bonds[0][4]} != {np.pi/2}"

    # check if the last particle bonds are correct
    selected_bonds = [bond for bond in bonds if (bond[0] == half_pipe.nparticles - 1 or bond[1] == half_pipe.nparticles - 1 or bond[2] == half_pipe.nparticles - 1)]
    assert len(selected_bonds) == 2, f"Particle {half_pipe.nparticles - 1} should be bonded with 2 particles, but found {len(selected_bonds)} bonds: {selected_bonds}"
    # check if this particles is bonded with nparticles - 2, nparticles - 3, nparticles - half_pipe.ny - 1 and nparticles - 2 * half_pipe.ny - 1
    assert (half_pipe.nparticles - 2) in [bond[0] for bond in selected_bonds] or (half_pipe.nparticles - 2) in [bond[1] for bond in selected_bonds] or (half_pipe.nparticles - 2) in [bond[2] for bond in selected_bonds], f"Particle {half_pipe.nparticles - 2} should be bonded with particle {half_pipe.nparticles - 1}, but not found in the bonds: {selected_bonds}"
    assert (half_pipe.nparticles - 3) in [bond[0] for bond in selected_bonds] or (half_pipe.nparticles - 3) in [bond[1] for bond in selected_bonds] or (half_pipe.nparticles - 3) in [bond[2] for bond in selected_bonds], f"Particle {half_pipe.nparticles - 3} should be bonded with particle {half_pipe.nparticles - 1}, but not found in the bonds: {selected_bonds}"
    assert (half_pipe.nparticles - half_pipe.ny - 1) in [bond[0] for bond in selected_bonds] or (half_pipe.nparticles - half_pipe.ny - 1) in [bond[1] for bond in selected_bonds] or (half_pipe.nparticles - half_pipe.ny - 1) in [bond[2] for bond in selected_bonds], f"Particle {half_pipe.nparticles - half_pipe.ny - 1} should be bonded with particle {half_pipe.nparticles - 1}, but not found in the bonds: {selected_bonds}"
    assert (half_pipe.nparticles - 2 * half_pipe.ny - 1) in [bond[0] for bond in selected_bonds] or (half_pipe.nparticles - 2 * half_pipe.ny - 1) in [bond[1] for bond in selected_bonds] or (half_pipe.nparticles - 2 * half_pipe.ny - 1) in [bond[2] for bond in selected_bonds], f"Particle {half_pipe.nparticles - 2 * half_pipe.ny - 1} should be bonded with particle {half_pipe.nparticles - 1}, but not found in the bonds: {selected_bonds}"

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



