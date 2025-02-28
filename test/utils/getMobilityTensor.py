import numpy as np

def getMobilityTensor(positions, solver):
    # Check if the solver is initialized

    # Check the dimensions of the positions array

    numberparticles = positions.shape[0]
    # Perform the algorithm to obtain the mobility tensor
    mobility_tensor = np.zeros((numberparticles*3, numberparticles*3))
    for i in range(numberparticles*3):
        force = np.zeros((numberparticles, 3))
        force[i//3,i-3*(i//3)] = 1
        velocity = solver.Mdot(force)[0]
        mobility_tensor[:, i] = velocity.reshape(-1, 1).flatten()

    # Return the mobility tensor as a matrix
    return mobility_tensor