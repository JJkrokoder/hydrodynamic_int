
from particles_mod.PlainMembrane import *

def test_membraneinit():
    """
    Test the initialization of the membrane.
    """
    membrane = Plain_Membrane()
    assert membrane.LengthX == 5.0
    assert membrane.LengthY == 5.0
    assert membrane.Density == 1.0
    assert membrane.Kp == 1.0
    assert membrane.Ka == 1.0
    assert membrane.nx == 5
    assert membrane.ny == 5
    assert membrane.nparticles == 25
    assert membrane.dx == 5.0/4
    assert membrane.dy == 5.0/4
    assert membrane.density == 1/(5.0/4 * 5.0/4)

def test_membrane_positions():
    """
    Test the generation of positions.
    """
    LengthX = 5.0
    LengthY = 6.0
    Density = 2.0
    membrane = Plain_Membrane(LengthX=LengthX, LengthY=LengthY, Density=Density)
    print(f"LengthX: {membrane.LengthX}, LengthY: {membrane.LengthY}, Density: {membrane.Density}")
    print(f"Number of particles: {membrane.nparticles}, Number of x positions: {membrane.nx}, Number of y positions: {membrane.ny}")
    positions = membrane.generate_positions()
    assert len(positions) == membrane.nparticles, f"Expected {membrane.nparticles} positions, got {len(positions)}"
    assert all(len(pos) == 3 for pos in positions), f"Expected each position to have 3 coordinates, got {[(len(pos), pos) for pos in positions if len(pos) != 3]}"
    assert all(-LengthX/2 <= pos[0] <= LengthX/2 for pos in positions), f"X coordinates out of bounds: {[(pos[0], pos) for pos in positions if not (-LengthX/2 <= pos[0] <= LengthX/2)]}"
    assert all(-LengthY/2 <= pos[1] <= LengthY/2 for pos in positions), f"Y coordinates out of bounds: {[(pos[1], pos) for pos in positions if not (-LengthY/2 <= pos[1] <= LengthY/2)]}"
    assert all(pos[2] == 0 for pos in positions), f"Z coordinates should be 0, got {[(pos[2], pos) for pos in positions if pos[2] != 0]}"

    first_column_idx = [i for i in range(len(positions)) if i % membrane.ny == 0]
    first_column_positions = [positions[i] for i in first_column_idx]
    assert all(pos[1] == -LengthY/2 for pos in first_column_positions), f"First column Y coordinates should be -LengthY/2, got {[pos[1] for pos in first_column_positions if pos[1] != -LengthY/2]}"
    last_column_idx = [i for i in range(len(positions)) if (i + 1) % membrane.ny == 0]
    last_column_positions = [positions[i] for i in last_column_idx]
    assert all(pos[1] == LengthY/2 for pos in last_column_positions), f"Last column Y coordinates should be LengthY/2, got {[pos[1] for pos in last_column_positions if pos[1] != LengthY/2]}"
    first_row_idx = [i for i in range(len(positions)) if i < membrane.ny]
    first_row_positions = [positions[i] for i in first_row_idx]
    assert all(pos[0] == -LengthX/2 for pos in first_row_positions), f"First row X coordinates should be -LengthX/2, got {[pos[0] for pos in first_row_positions if pos[0] != -LengthX/2]}"
    last_row_idx = [i for i in range(len(positions)) if i >= (membrane.nparticles - membrane.ny)]
    last_row_positions = [positions[i] for i in last_row_idx]
    assert all(pos[0] == LengthX/2 for pos in last_row_positions), f"Last row X coordinates should be LengthX/2, got {[pos[0] for pos in last_row_positions if pos[0] != LengthX/2]}"
    
    assert all(pos[1] == -LengthY/2 + (i % membrane.ny) * membrane.dy for i, pos in enumerate(positions)), f"Y coordinates are incorrect, got {[pos[1] for i, pos in enumerate(positions) if pos[1] != -LengthY/2 + (i % membrane.ny) * membrane.dy]}"
    assert all(pos[0] == -LengthX/2 + (i // membrane.ny) * membrane.dx for i, pos in enumerate(positions)), f"X coordinates are incorrect, got {[pos[0] for i, pos in enumerate(positions) if pos[0] != -LengthX/2 + (i // membrane.ny) * membrane.dx]}"


def test_membrane_pairbonds():
    """
    Test the generation of pair bonds.
    """
    LengthX = 5.0
    LengthY = 6.0
    Density = 2.0
    Kspring = 2.0
    membrane = Plain_Membrane(LengthX=LengthX, LengthY=LengthY, Density=Density, Kp = Kspring)

    assert membrane.dy == LengthY/(membrane.ny - 1)
    assert membrane.dx == LengthX/(membrane.nx - 1)


    positions = membrane.generate_positions()
    pairbonds = membrane.generate_pairbonds(positions)
    num_horizontal_bonds = (membrane.nx) * (membrane.ny - 1)
    num_diagonal_bonds = 2*(membrane.nx - 1) * (membrane.ny - 1)
    num_vertical_bonds = (membrane.ny) * (membrane.nx - 1)
    num_pairbonds = num_horizontal_bonds + num_vertical_bonds + num_diagonal_bonds

    assert len(pairbonds) == num_pairbonds, f"Expected {num_pairbonds} pair bonds, got {len(pairbonds)}"

    horizontal_bonds = [bond for bond in pairbonds if np.isclose(positions[bond[0]][0], positions[bond[1]][0], atol=1e-6)]
    vertical_bonds = [bond for bond in pairbonds if np.isclose(positions[bond[0]][1], positions[bond[1]][1], atol=1e-6)]
    diagonal_bonds = [bond for bond in pairbonds if not (np.isclose(positions[bond[0]][0], positions[bond[1]][0], atol=1e-6) or np.isclose(positions[bond[0]][1], positions[bond[1]][1], atol=1e-6))]
    
    diagonal_right_bonds = [bond for bond in diagonal_bonds if (np.isclose(positions[bond[1]][1] - positions[bond[0]][1], membrane.dy, atol=1e-6))]
    diagonal_left_bonds = [bond for bond in diagonal_bonds if (np.isclose(positions[bond[0]][1] - positions[bond[1]][1], membrane.dy, atol=1e-6))]
                                                   
    assert len(horizontal_bonds) == num_horizontal_bonds, f"Expected {num_horizontal_bonds} horizontal bonds, got {len(horizontal_bonds)}"
    assert len(vertical_bonds) == num_vertical_bonds, f"Expected {num_vertical_bonds} vertical bonds, got {len(vertical_bonds)}"
    assert len(diagonal_bonds) == num_diagonal_bonds, f"Expected {num_diagonal_bonds} diagonal bonds, got {len(diagonal_bonds)}"

    assert len(diagonal_right_bonds) == num_diagonal_bonds/2, f"Expected {num_diagonal_bonds/2} diagonal right bonds, got {len(diagonal_right_bonds)}"
    assert len(diagonal_left_bonds) == num_diagonal_bonds/2, f"Expected {num_diagonal_bonds/2} diagonal left bonds, got {len(diagonal_left_bonds)}"

    for bond in horizontal_bonds:
        assert bond[1] - bond[0] == 1, f"Horizontal bond should be adjacent, got {bond[1] - bond[0]}"
    for bond in vertical_bonds:
        assert bond[1] - bond[0] == membrane.ny, f"Vertical bond should be adjacent, got {bond[1] - bond[0]}"
    for bond in diagonal_bonds:
        assert bond[1] - bond[0] == membrane.ny + 1 or bond[1] - bond[0] == membrane.ny - 1, f"Diagonal bond should be adjacent, got {bond[1] - bond[0]}"
        assert np.isclose(bond[3],np.sqrt(membrane.dx**2 + membrane.dy**2), atol=1e-6), f"Diagonal bond distance should be sqrt(dx^2 + dy^2), got {bond[3]}"
    for bond in diagonal_right_bonds:
        assert bond[1] - bond[0] == membrane.ny + 1, f"Diagonal right bond should be adjacent, got {bond[1] - bond[0]}"
    for bond in diagonal_left_bonds:
        assert bond[1] - bond[0] == membrane.ny - 1, f"Diagonal left bond should be adjacent, got {bond[1] - bond[0]}"

def test_membrane_anglebonds():
    """
    Test the generation of angular bonds.
    """
    LengthX = 5.0
    LengthY = 6.0
    Density = 2.0
    Kangle = 2.0
    membrane = Plain_Membrane(LengthX=LengthX, LengthY=LengthY, Density=Density, Ka = Kangle)

    assert membrane.dy == LengthY/(membrane.ny - 1)
    assert membrane.dx == LengthX/(membrane.nx - 1)

    positions = membrane.generate_positions()
    angularbonds = membrane.generate_anglebonds(positions)

    num_horizontal_bonds = (membrane.nx) * (membrane.ny - 2)
    num_vertical_bonds = (membrane.ny) * (membrane.nx - 2)
    num_angular_bonds = num_horizontal_bonds + num_vertical_bonds

    horizontal_bonds = [bond for bond in angularbonds if np.isclose(positions[bond[0]][0], positions[bond[1]][0], atol=1e-6)]
    vertical_bonds = [bond for bond in angularbonds if np.isclose(positions[bond[0]][1], positions[bond[1]][1], atol=1e-6)]

    assert len(horizontal_bonds) == num_horizontal_bonds, f"Expected {num_horizontal_bonds} horizontal bonds, got {len(horizontal_bonds)}"
    assert len(vertical_bonds) == num_vertical_bonds, f"Expected {num_vertical_bonds} vertical bonds, got {len(vertical_bonds)}"
    assert len(angularbonds) == num_angular_bonds, f"Expected {num_angular_bonds} angular bonds, got {len(angularbonds)}"

    for bond in angularbonds:
        assert np.isclose(bond[4], np.pi, atol=1e-6), f"Angular bond should be 180 degrees, got {bond[4]}"
    for bond in horizontal_bonds:
        assert bond[1] - bond[0] == 1, f"Horizontal bond should be adjacent, got {bond[1] - bond[0]}"
        assert bond[2] - bond[1] == 1, f"Horizontal bond should be adjacent, got {bond[2] - bond[1]}"
    for bond in vertical_bonds:
        assert bond[1] - bond[0] == membrane.ny, f"Vertical bond should be adjacent, got {bond[1] - bond[0]}"
        assert bond[2] - bond[1] == membrane.ny, f"Vertical bond should be adjacent, got {bond[2] - bond[1]}"

