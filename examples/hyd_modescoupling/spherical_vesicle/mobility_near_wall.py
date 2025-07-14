import numpy as np
import matplotlib.pyplot as plt
import particles_mod.Icosphere as ico
import libMobility as lm


def create_solver(positions: np.ndarray, wallHeight: float = 0.0,
                  hydrodynamicRadius: float = 1.0, viscosity: float = 1/(6 * np.pi)):
    """
    Create a libmobility solver based on the NBodywall mobility.
    Args:
        positions: The positions of the particles.
        wallHeight: Height of the wall for the NBodywall model. Default is 0.0.
    Returns:
        solver: An instance of the solver for the specified model.
    """

    solver = lm.NBody("open", "open", "single_wall")
    solver.setParameters(wallHeight = wallHeight)

    solver.initialize(
        temperature=0.0,
        viscosity=viscosity,
        hydrodynamicRadius=hydrodynamicRadius,
    )

    solver.setPositions(positions)
    return solver

min_radius = 10.0  # Minimum radius of the sphere
max_radius = 15.0  # Maximum radius of the sphere

sphere_radius = np.linspace(min_radius, max_radius, 5)
density = 1/(np.pi * 1.05**2)  # Closed package distribution

num_distances = 30
limit_diameter = 10  # Limit for the distance to the wall in terms of the sphere diameter

tt_eff_mobility = np.zeros((len(sphere_radius), num_distances))
rt_eff_mobility = np.zeros((len(sphere_radius), num_distances))
all_distances = np.zeros((len(sphere_radius), num_distances))

for j, radius in enumerate(sphere_radius):
    print(f"Calculating for radius {radius:.2f}")
    icosphere = ico.IcoSphere(radius=radius, density=density, Kpair=2, Kdi=2)
    icosphere.obtain_modes()
    icosphere.modes = ico.reconstruct_modes(modes=icosphere.modes, positions=icosphere.positions)
    distances = np.logspace(-3, np.log10(limit_diameter*2*radius), num_distances)
    all_distances[j, :] = distances
    for i, distance in enumerate(distances):
        solver = create_solver(icosphere.positions, wallHeight=-radius - distance)
        coupling_matrix = icosphere.obtain_normal_coupled_mobility(solver)
        N_matrix = coupling_matrix[6:,6:]
        N_inverse = np.linalg.pinv(N_matrix)

        normalized_displacements = N_inverse @ coupling_matrix[6:,0]

        tt_coupling = coupling_matrix[0,0] - coupling_matrix[0,6:] @ normalized_displacements
        rt_coupling = coupling_matrix[4,0] - coupling_matrix[4,6:] @ normalized_displacements

        linear_force = 1.0 
        linear_force_coeff = linear_force * icosphere.modes[0,0]

        rotational_velocity = np.linalg.norm(icosphere.modes[0,:3]) / radius

        tt_velocity_coeff = tt_coupling * linear_force_coeff
        rt_velocity_coeff = rt_coupling * linear_force_coeff

        tt_velocity = tt_velocity_coeff * icosphere.modes[0,0]
        rt_velocity = np.linalg.norm(rt_velocity_coeff * icosphere.modes[0,:3]) / radius

        tt_eff_mobility[j, i] = tt_velocity / linear_force
        rt_eff_mobility[j, i] = rt_velocity / linear_force


print("Calculation complete.")

# Save results to file

np.savez("mobility_near_wall.npz",
         sphere_radius=sphere_radius,
         all_distances=all_distances,
         tt_eff_mobility=tt_eff_mobility,
         rt_eff_mobility=rt_eff_mobility)






