import numpy as np

def getMobilityTensor(positions, solver):
    '''
    This function calculates the mobility tensor of a system of particles given their positions and a solver object
    
    Parameters
    ----------
    positions: numpy array
        The positions of the particles in the system    
    solver: SelfMobility object
        The solver object used to calculate the mobility tensor
    
    Returns
    -------
    mobility_tensor: numpy array
        The mobility tensor of the system
    '''

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