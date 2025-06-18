import hydrodynamic_int.utils as hydint
import libMobility as lm
import numpy as np
from pytest import mark

def create_default_solver(method="SelfMobility"):
    """Create a default solver for testing purposes."""
    if method == "SelfMobility":
        solver = lm.SelfMobility("open", "open", "open")
        solver.setParameters(5)
    else:
        solver = lm.NBody("open", "open", "open")
        solver.setParameters()
    solver.initialize(
        temperature=0,
        viscosity=1/(6 * np.pi),  # Viscosity for a sphere in a fluid
        hydrodynamicRadius=1.0,  # Default hydrodynamic radius
        needsTorque=False,
    )
    return solver
    
    
@mark.parametrize("numberparticles", [1, 10])
@mark.parametrize("method", ["SelfMobility", "NBody"])
def test_Mobility_symmetry(numberparticles, method):
    """Test that the mobility tensor is symmetric."""
    solver = create_default_solver(method)
    positions = np.random.rand(numberparticles, 3)
    mobility_tensor = hydint.getMobilityTensor(positions, solver)
    assert np.allclose(mobility_tensor, mobility_tensor.T, atol=1e-8, rtol=0), \
           "Mobility tensor should be symmetric."  


@mark.parametrize("numberparticles", [1, 10])
def test_selfMobility_matrix(numberparticles):
    """Test that the self mobility matrix for a single particle at the origin is identity."""
    solver = create_default_solver()
    positions = np.random.rand(numberparticles, 3)
    mobility_tensor = hydint.getMobilityTensor(positions, solver)
    assert np.all(mobility_tensor == np.eye(numberparticles * 3)), \
              "Mobility tensor for single particle at origin should be identity matrix."


@mark.parametrize("distance_log", range(6))
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
    
