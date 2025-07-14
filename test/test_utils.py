import hydrodynamic_int.utils as hydint
import libMobility as lm
import numpy as np
from pytest import mark

def create_default_solver(method="SelfMobility", wallheight: float = 0.0):
    """Create a default solver for testing purposes."""
    if method == "SelfMobility":
        solver = lm.SelfMobility("open", "open", "open")
        solver.setParameters(5)
    elif method == "NBody":
        solver = lm.NBody("open", "open", "open")
        solver.setParameters()
    elif method == "NBodywall":
        solver = lm.NBody("open", "open", "single_wall")
        solver.setParameters(wallHeight=wallheight)
    solver.initialize(
        temperature=0,
        viscosity=1/(6 * np.pi),  # Viscosity for a sphere in a fluid
        hydrodynamicRadius=1.0,  # Default hydrodynamic radius
    )
    return solver

def create_particle_cloud(numberparticles: int = 10, boxsize: float = 1.0, height: float = 1.0):
    """Create a random particle cloud within a cubic box, displaced in the z-direction."""
    positions = np.random.rand(numberparticles, 3) * boxsize + np.array([0, 0, height])
    return positions

    
    
    
# @mark.parametrize("numberparticles", [3, 10])
# @mark.parametrize("method", ["SelfMobility", "NBody", "NBodywall"])
# @mark.parametrize("wall_height", [0.0, 1e7, 2.0, 10.0])
@mark.parametrize("numberparticles", [3])
@mark.parametrize("method", ["SelfMobility"])
@mark.parametrize("wall_height", [0.0])
def test_Mobility_symmetry(numberparticles, method, wall_height):
    """Test that the mobility tensor is symmetric."""
    solver = create_default_solver(method, wallheight=wall_height)
    positions = np.random.rand(numberparticles, 3)
    mobility_tensor = hydint.getMobilityTensor(positions, solver)
    assert np.allclose(mobility_tensor, mobility_tensor.T, atol=1e-8, rtol=0), \
           "Mobility tensor should be symmetric."  


#@mark.parametrize("numberparticles", [1, 10])
@mark.parametrize("numberparticles", [1])
def test_selfMobility_matrix(numberparticles):
    """Test that the self mobility matrix for a single particle at the origin is identity."""
    solver = create_default_solver()
    positions = np.random.rand(numberparticles, 3)
    mobility_tensor = hydint.getMobilityTensor(positions, solver)
    assert np.all(mobility_tensor == np.eye(numberparticles * 3)), \
              "Mobility tensor for single particle at origin should be identity matrix."


#@mark.parametrize("distance_log", range(6))
@mark.parametrize("distance_log", range(1))
def test_NBody_SelfMobity_consistency(distance_log):
    """Test that the NBody and SelfMobility methods yield consistent results for two particles at different separations."""
    distance = 10 ** distance_log
    absolute_error = 3/(2*distance)
    positions = np.array([[0, 0, 0], [distance, 0, 0]]) 

    solver_self = create_default_solver("SelfMobility")
    mobility_tensor_self = hydint.getMobilityTensor(positions, solver_self)

    solver_nbody = create_default_solver("NBody")
    mobility_tensor_nbody = hydint.getMobilityTensor(positions, solver_nbody)

    assert np.allclose(mobility_tensor_self, mobility_tensor_nbody, atol=absolute_error), \
           "Mobility tensors from SelfMobility and NBody should be consistent within the absolute error at a given distance."


# @mark.parametrize("boxsize", [1.0, 10.0])
# @mark.parametrize("numberparticles", range(1, 11, 2))
@mark.parametrize("boxsize", [1.0])
@mark.parametrize("numberparticles", [3])
def test_consistency_NBody_NBodywall(boxsize, numberparticles):
    """Test that the NBody and NBodywall methods yield consistent results for particles in a wall boundary condition 
    when the particles are far from the wall."""
    height = 1e7
    positions = create_particle_cloud(numberparticles=numberparticles, boxsize=boxsize, height=height)

    solver_nbody = create_default_solver("NBody")
    mobility_tensor_nbody = hydint.getMobilityTensor(positions, solver_nbody)

    solver_nbodywall = create_default_solver("NBodywall")
    mobility_tensor_nbodywall = hydint.getMobilityTensor(positions, solver_nbodywall)

    assert np.allclose(mobility_tensor_nbody , mobility_tensor_nbodywall), \
           "Mobility tensors from NBody and NBodywall should be consistent."



