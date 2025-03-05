import numpy as np
import matplotlib.pyplot as plt
from libMobility import NBody
from hydrodynamic_int import getMobilityTensor

def test_RPY_distance():
    '''
    Test the RPY mobility tensor distance dependence for a system of two particles with NBody solver.

    Parameters
    ----------
    None

    Returns
    -------
    None
    '''

    # Define the number of particles in the system
    numberParticles = 2
    # Coordinates of the system
    coordinates =['x', 'y', 'z']
    # Define the viscosity and hydrodynamic radius
    viscosity = 1.0
    hydrodynamicRadius = 1.0
    # Set calculation precision based on solver's precision
    precision = np.float64
    # STEP 1: Create solver object with open boundary conditions in all three dimensions
    solver = NBody("open", "open", "open")
    # STEP 2: Configure specific solver parameters
    solver.setParameters(algorithm="advise", Nbatch=1, NperBatch=numberParticles)
    # STEP 3: Initialize solver with global parameters
    solver.initialize(
      temperature=0.0,
      viscosity=viscosity,
      hydrodynamicRadius=hydrodynamicRadius,
      numberParticles=numberParticles
    )

    #Now I need to create a stricture that allows me to calculate de mobility for a set of distances (maybe with a function outside or somewhat)
    # Create a set of distances to calculate the mobility tensor
    distances = np.linspace(0.1, 10, 100)
    # Create a list to store the mobility tensor for each distance
    mobility_tensors = []
    # Create a position array
    positions = np.zeros((numberParticles, 3), dtype=precision)
    # Loop over the distances
    for distance in distances:
        # Set position of particle 2
        positions[1, 0] = distance
        # Set the position of the particles
        solver.setPositions(positions)
        # Obtain the mobility tensor
        mobility_tensors.append(getMobilityTensor(positions, solver))

    # Mobility tensor array
    mobility_tensors = np.array(mobility_tensors)
    # dimension assertion for mobility tensor matrix (3Nx3N)
    assert mobility_tensors.shape[1] == 3*numberParticles
    # square matrix assertion
    assert mobility_tensors.shape[1] == mobility_tensors.shape[2]
    # symmetric matrix assertion
    assert np.allclose(mobility_tensors, mobility_tensors.transpose(0, 2, 1))

    # Now we need to check that the diagonal 3x3 blocks of the total tensor are equal to a 
    # diagonal 3x3 selfmobility matrix
    #
    # Create a 3x3 selfmobility matrix
    selfmobility_scalar = 1/(6*np.pi*viscosity*hydrodynamicRadius)
    selfmobility = np.eye(3)*selfmobility_scalar
    #Check if each of the diagonal blocks is close enough to the selfmobility matrix
    for i in range(len(distances)):
        for j in range(numberParticles):
            assert np.allclose(mobility_tensors[i, 3*j:3*j+3, 3*j:3*j+3], selfmobility, rtol=1e-3, atol=1e-6)

    # Check if the non diagonal elements of the cross mobility tensor are close to zero
    for i in range(len(distances)): # Loop over the distances
        for j in range(numberParticles-1): # Loop over the particles
            for k in range(j+1, numberParticles): # Loop over the particles
                    # Loop over the coordinates
                    for l in range(3):
                        for m in range(l+1,3):
                            assert np.allclose(mobility_tensors[i, 3*j+l, 3*k+m], 0, rtol=1e-3, atol=1e-6)
                            assert np.allclose(mobility_tensors[i, 3*j+m, 3*k+l], 0, rtol=1e-3, atol=1e-6)

    # Check if the perpendicular diagonal elements of the cross mobility tensor are equal (or close enough)
    for i in range(len(distances)): # Loop over the distances
        for j in range(numberParticles-1): # Loop over the particles
            for k in range(j+1, numberParticles): # Loop over the particles
                for l in range(1,3):
                    assert np.allclose(mobility_tensors[i, 3*j+l, 3*k+l], mobility_tensors[i, 3*j+l, 3*k+l], rtol=1e-3, atol=1e-6)

    # Plot the distance dependence of the mobility tensor non diagonal blocks elements
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    for i in range(3): # Loop over the coordinates
        for j in range(i, 3): # Loop over the coordinates
                ax.plot(distances, mobility_tensors[:, i, 3+j]/selfmobility_scalar, label="$\\mathcal{M}_{"+coordinates[i]+coordinates[j]+"}$")
    # Set the font size
    plt.rcParams.update({'font.size': 16})
    # Set the labels
    ax.set_xlabel('x/a')
    ax.set_ylabel('Mobility element ($\\mathcal{M}_0$)')
    ax.xaxis.label.set_fontsize(16)
    ax.yaxis.label.set_fontsize(16)
    ax.legend()
    # Set the ticks
    ax.tick_params(axis='both', which='major', labelsize=16)
    plt.tight_layout()
    # Save the figure
    plt.savefig('mobility_tensor_distance_dependence.png')

    
test_RPY_distance()
    






