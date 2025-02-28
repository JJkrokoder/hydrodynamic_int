import numpy as np
from libMobility import SelfMobility

'''
This file calculates the linear and angular velocities of a system of particles as a
result of the forces and torques acting on them.
The conection between these quatities is mediated by the Self Mobility of the system.
The calculation is done using the Mdot method (M·[F,T]), by utilising the SelfMobility
class of the libMobility library.

The example demonstrates the following steps:
1. Create a SelfMobility object with open boundary conditions in all three dimensions.
2. Configure the solver with specific parameters.
3. Initialize the solver with global parameters.
4. Set the positions of the particles in the solver.
5. Calculate the linear and angular velocities of the particles using the Mdot method.

The example uses random values for the positions, forces, and torques of the particles
and prints the resulting linear and angular velocities.

Parameters
----------
precision : np.float32 or np.float64
    Precision of the calculations.
boundaryConditions : str
    Boundary conditions of the system for each dimension.
parameter : int
    Specific parameter for the solver.
globalParameters : dict
    Global parameters for the solver.
    temperature : float
        Temperature of the system.
    viscosity : float
        Viscosity of the fluid.
    hydrodynamicRadius : float
        Hydrodynamic effective radius of the particles.
    numberParticles : int
        Number of particles in the system.
    needsTorque : bool
        Flag to specify the necessity of torque calculations.
positions : np.ndarray
    Positions of the particles in the system. Dimension: (numberParticles, 3).
forces : np.ndarray
    Forces acting on the particles. Dimension: (numberParticles, 3).
torques : np.ndarray
    Torques acting on the particles. Dimension: (numberParticles, 3).

Returns
-------
linear : np.ndarray
    Linear velocities of the particles. Dimension: (numberParticles, 3).
angular : np.ndarray
    Angular velocities of the particles. Dimension: (numberParticles, 3).

Example
-------
>>> python SelfMobility_1.py
'''

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