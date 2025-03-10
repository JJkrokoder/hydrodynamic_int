import numpy as np
import matplotlib.pyplot as plt
from libMobility import NBody
from hydrodynamic_int import getMobilityTensorRPY
import os
from datetime import datetime

def plotMobilityTensor(mobility_tensor, distances, coordinates):
    '''
    Plot the mobility tensor distance dependence for a pair of particles.

    Parameters
    ----------
    mobility_tensor : np.array
        Mobility tensor array for differente distances. It should be normalized by the selfmobility scalar.
    distances : np.array
        Array of distances. They should be normalized by the hydrodynamic radius.
    coordinates : list
        List of coordinates.

    Returns
    -------
    None
    '''

    # Check if the mobility tensor array dimension [0] is equal to the number of distances
    assert mobility_tensor.shape[0] == len(distances)

    # Plot the distance dependence of the mobility tensor non diagonal blocks elements
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    for i in range(len(coordinates)): # Loop over the coordinates
        ax.plot(distances, mobility_tensor[:, i, 3+i], label="$\\mathcal{M}_{"+coordinates[i]+coordinates[i]+"}$")
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
    # Create the output directory if it does not exist
    if not os.path.exists('output'):
        os.makedirs('output')
    # Save the figure
    plt.savefig('output/RPY_mobility_tensor_distance_dependence_.png')

def get_theCrossMobRPY(distances):
    '''
    Calculate the normalized RPY cross mobility tensor for a pair of particles with RPY mobility tensor.

    Parameters
    ----------
    distances : np.array
        Distances between the particlesm normalized by the hydrodynamic radius.

    Returns
    -------
    cross_mobility : np.array
        RPY cross mobility tensor for a pair of particles at different distances.
    '''

    cross_mobility = np.zeros((len(distances),3, 3))
    # Calculate the cross mobility tensor
    for i, distance in enumerate(distances):
        if distance <= 2:
            for j in range(3):
                cross_mobility[i, j, j] = 1 - 9/32*distance
            cross_mobility[i, 0, 0] += 3/32*distance # Add the x component
        else:
            for j in range(3):
                cross_mobility[i, j, j] = 3/(4*distance) + 1/(2*distance**3)
            cross_mobility[i, 0, 0] += 3/(4*distance) - 3/(2*distance**3) # Add the x component
    
    return cross_mobility
    

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
    # Define the hydrodynamic radius
    hydrodynamicRadius = 1.0
    # Define the viscosity
    viscosity = 1.0
    # Define the precision
    precision = np.float64

    # Create a set of distances to calculate the mobility tensor
    distances = np.linspace(0.1, 10, 100)*hydrodynamicRadius
    # Create a list to store the mobility tensor for each distance
    mobility_tensors = []
    # Create a position array
    positions = np.zeros((numberParticles, 3), dtype=precision)
    # Loop over the distances
    for distance in distances:
        # Set position of particle 2
        positions[1, 0] = distance
        # Obtain the mobility tensor
        mobility_tensors.append(getMobilityTensorRPY(positions, hyd_radius=hydrodynamicRadius, viscosity=viscosity))

    # Mobility tensor array
    mobility_tensors = np.array(mobility_tensors)

    # dimension assertion for mobility tensor matrix size
    assert mobility_tensors.shape[1] == 3*numberParticles
    # square matrix assertion
    assert mobility_tensors.shape[1] == mobility_tensors.shape[2]
    # symmetric matrix assertion
    assert np.allclose(mobility_tensors, mobility_tensors.transpose(0, 2, 1))

    # Check that the diagonal 3x3 blocks of the total tensor are equal to a selfmobility matrix
    selfmobility_scalar = 1/(6*np.pi*viscosity*hydrodynamicRadius) # Create a 3x3 selfmobility matrix
    selfmobility = np.eye(3)*selfmobility_scalar
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

    # Plot the mobility tensor distance dependence
    # plotMobilityTensor(mobility_tensors/selfmobility_scalar, distances/hydrodynamicRadius, coordinates)

    # Check if the cross mobility block is equal to the theoretical matrix

    # Theoretical cross mobility matrix
    cross_mobility_the = get_theCrossMobRPY(distances/hydrodynamicRadius)*selfmobility_scalar
    # Check the cross mobility tensor
    for i in range(len(distances)):
        for j in range(3):
            for k in range(3):
                assert np.allclose(mobility_tensors[i, j, 3+k], cross_mobility_the[i, j, k], rtol=1e-3, atol=1e-6)


    










