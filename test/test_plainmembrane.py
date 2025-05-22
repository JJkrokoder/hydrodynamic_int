
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





