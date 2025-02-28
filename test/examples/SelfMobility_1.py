import numpy as np
from libMobility import SelfMobility

# Define the number of particles in the system
numberParticles = 5

# Set calculation precision based on solver's precision
precision = np.float32 if SelfMobility.precision == "float" else np.float64

# STEP 1: Create solver object with open boundary conditions in all three dimensions
solver = SelfMobility("open", "open", "open")

# STEP 2: Configure specific solver parameters
solver.setParameters(parameter=5) # SelfMobility only exposes an example parameter

# STEP 3: Initialize solver with global parameters
solver.initialize(
  temperature=0.0,         # No thermal effects (deterministic)
  viscosity=1.0,           # Fluid viscosity
  hydrodynamicRadius=1.0,  # Hydrodynamic radius of particles
  numberParticles=numberParticles,
  needsTorque=True,        # Enable torque calculations
)

# Create random positions for particles
positions = np.random.rand(numberParticles, 3).astype(precision)

# Define forces and torques acting on particles
forces = np.random.rand(numberParticles, 3).astype(precision)
torques = np.random.rand(numberParticles, 3).astype(precision)

# Set particle positions in the solver
solver.setPositions(positions)

# Calculate linear and angular velocities using Mdot method (M·[F,T])
linear, angular = solver.Mdot(forces, torques)

# Print linear and angular velocities
print("Linear Velocities:")
print(linear)
print("Angular Velocities:")
print(angular)