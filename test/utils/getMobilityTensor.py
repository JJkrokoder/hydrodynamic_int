import numpy as np

def getMobilityTensor(positions, solver):
    # Check if the solver is initialized

    # Check the dimensions of the positions array
    if len(positions) != solver.numberParticles*3:
        raise ValueError("Number of particles in positions array does not match the solver.")

    # Perform the algorithm to obtain the mobility tensor
    mobility_tensor = np.zeros((solver.numberParticles*3, solver.numberParticles*3))
    for i in range(solver.numberParticles*3):
        force = np.zeros(solver.numberParticles*3)
        force[i] = 1
        velocity = solver.Mdot(force)
        mobility_tensor[:, i] = velocity

    # Return the mobility tensor as a matrix
    return mobility_tensor