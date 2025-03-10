import numpy as np
import libMobility as lb

def getMobilityTensor(positions, solver):
    '''
    This function calculates the mobility tensor of a system of particles given their positions and a solver object.
    The solver must be initialized before calling this function.
    
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

def getMobilityTensorRPY(positions, hyd_radius = 1.0, viscosity = 1.0, boundary_conditions = ['open', 'open', 'open']):
    '''
    This function calculates the mobility tensor (NBody solver) of a system of particles given their positions and hydrodynamic parameters.
    
    Parameters
    ----------
    positions: numpy array
        The positions of the particles in the system
    hyd_radius: float
        The hydrodynamic radius of the particles
    viscosity: float
        The viscosity of the fluid
    boundary_conditions: list
        The boundary conditions of the system   
    
    
    Returns
    -------
    mobility_tensor: numpy array
        The mobility tensor of the system
    '''
    # Solver initialization
    numberparticles = positions.shape[0]
    solver = lb.NBody(boundary_conditions[0], boundary_conditions[1], boundary_conditions[2])
    solver.setParameters(algorithm="advise", Nbatch=1, NperBatch=positions.shape[0])
    solver.initialize(
        temperature=0.0,
        viscosity=viscosity,
        hydrodynamicRadius=hyd_radius,
        numberParticles=positions.shape[0]
    )
    solver.setPositions(positions)

    # Perform the algorithm to obtain the mobility tensor
    mobility_tensor = np.zeros((numberparticles*3, numberparticles*3))
    for i in range(numberparticles*3):
        force = np.zeros((numberparticles, 3))
        force[i//3,i-3*(i//3)] = 1
        velocity = solver.Mdot(force)[0]
        mobility_tensor[:, i] = velocity.reshape(-1, 1).flatten()

    # Return the mobility tensor as a matrix
    return mobility_tensor