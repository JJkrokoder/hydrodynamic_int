
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
    positions = membrane.generate_positions()
    assert len(positions) == membrane.nparticles, f"Expected {membrane.nparticles} positions, got {len(positions)}"
    assert all(len(pos) == 3 for pos in positions), f"Expected each position to have 3 coordinates, got {[(len(pos), pos) for pos in positions if len(pos) != 3]}"
    assert all(-LengthX/2 <= pos[0] <= LengthX/2 for pos in positions), f"X coordinates out of bounds: {[(pos[0], pos) for pos in positions if not (-LengthX/2 <= pos[0] <= LengthX/2)]}"
    assert all(-LengthY/2 <= pos[1] <= LengthY/2 for pos in positions), f"Y coordinates out of bounds: {[(pos[1], pos) for pos in positions if not (-LengthY/2 <= pos[1] <= LengthY/2)]}"
    assert all(pos[2] == 0 for pos in positions), f"Z coordinates should be 0, got {[(pos[2], pos) for pos in positions if pos[2] != 0]}"

    




