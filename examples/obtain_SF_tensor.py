import numpy as np
from libMobility import SelfMobility
from hydrodynamic_int import getMobilityTensor

# Define the number of particles in the system
numberParticles = 2
# Set calculation precision based on solver's precision
precision = np.float64
# STEP 1: Create solver object with open boundary conditions in all three dimensions
solver = SelfMobility("open", "open", "open")
# STEP 2: Configure specific solver parameters
solver.setParameters(parameter=5)
# STEP 3: Initialize solver with global parameters
solver.initialize(
  temperature=0.0,
  viscosity=1.0,
  hydrodynamicRadius=1.0,
  numberParticles=numberParticles,
  needsTorque=False,
)

# Create random positions for particles
positions = np.random.rand(numberParticles, 3).astype(precision)

# Obtain the mobility tensor
mobilitytensor=getMobilityTensor(positions, solver)
print(mobilitytensor)
print(' ')

# Theoretical Tensor
theoreticalTensor = np.diag([1 / (6 * np.pi)] * (3 * numberParticles))
print(theoreticalTensor)