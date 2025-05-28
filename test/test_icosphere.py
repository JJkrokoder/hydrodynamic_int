import particles_mod.Icosphere as ico
import numpy as np

def test_icosphere():
    
    r = 2.0
    d = 3.0
    Kpair = 0.5

    nparticles = round(4 * np.pi * r**2 * d)
    
    icosphere = ico.IcoSphere(radius=r, density=d, Kpair=Kpair)

    assert icosphere.radius == r, f"Expected radius {r}, got {icosphere.radius}"
    assert icosphere.Kpair == Kpair, f"Expected Kpair {Kpair}, got {icosphere.Kpair}"
    assert icosphere.Kangle == 1.0, f"Expected Kangle 1.0, got {icosphere.Kangle}"

    assert np.isclose(icosphere.nparticles, nparticles, rtol=1e-1), f"Expected nparticles {nparticles}, got {icosphere.nparticles}"
    assert np.isclose(icosphere.density, d, rtol=1e-2), f"Expected density {d}, got {icosphere.density}"

    distances = np.linalg.norm(icosphere.positions, axis=1)
    assert np.all(np.isclose(distances, r, atol=1e-12)), "Not all particles are at the expected radius"

def test_icosphere_positions():
    r = 20.0
    d = 4.0

    icosphere = ico.IcoSphere(radius=r, density=d)
    positions = icosphere.generate_positions()

    assert isinstance(positions, list), "Positions should be a list"
    assert all(isinstance(pos, list) for pos in positions), "Each position should be a list"

    assert len(positions) == icosphere.nparticles, f"Expected {icosphere.nparticles} positions, got {len(positions)}"
    
    for pos in positions:
        assert len(pos) == 3, f"Position {pos} does not have 3 coordinates"
        assert np.isclose(np.linalg.norm(pos), r, atol=1e-12), f"Position {pos} is not at the expected radius {r}"


