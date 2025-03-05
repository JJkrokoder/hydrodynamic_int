import numpy as np
from libMobility import NBody
from hydrodynamic_int import getMobilityTensor

def test_RPY():
    '''
    Test the RPY mobility tensor for a system of two particles with NBody solver and specific positions.
    The positions are defined below and the mobility tensor is calculated using the getMobilityTensor function.
    At the end, the mobility tensor is printed.

    Parameters
    ----------
    None

    Returns
    -------
    None
    '''

    # Define the number of particles in the system
    numberParticles = 2
    # Set calculation precision based on solver's precision
    precision = np.float64
    # STEP 1: Create solver object with open boundary conditions in all three dimensions
    solver = NBody("open", "open", "open")
    # STEP 2: Configure specific solver parameters
    solver.setParameters(algorithm="advise", Nbatch=1, NperBatch=numberParticles)
    # STEP 3: Initialize solver with global parameters
    solver.initialize(
      temperature=0.0,
      viscosity=1.0,
      hydrodynamicRadius=1.0,
      numberParticles=numberParticles
    )

    # Create position where partircles are separated by 10 unit in x direction
    positions = np.array([[0, 0, 0], [10, 0, 0]], dtype=precision)

    # Set the position of the particles
    solver.setPositions(positions)

    # Obtain the mobility tensor
    mobilitytensor=getMobilityTensor(positions, solver)
