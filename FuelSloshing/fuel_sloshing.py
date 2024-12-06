import numpy as np

def compute_sloshing_forces_moments(theta, phi, psi, omega, omega_dot, delta_s, delta_s_dot, params):
    """
    Computes sloshing forces and moments using advanced rotary sloshing equations,
    adjusted for the offset between the tank pivot and the rocket's CoG.

    Parameters:
        theta (float): Pitch angle (rad).
        phi (float): Roll angle (rad).
        psi (float): Yaw angle (rad).
        omega (numpy.ndarray): Angular velocity [omega_x, omega_y, omega_z] (rad/s).
        omega_dot (numpy.ndarray): Angular acceleration [omega_x_dot, omega_y_dot, omega_z_dot] (rad/s^2).
        delta_s (numpy.ndarray): Sloshing mass displacement vector [y_s, z_s] (m).
        delta_s_dot (numpy.ndarray): Sloshing mass velocity vector [y_dot_s, z_dot_s] (m/s).
        params (dict): Dictionary containing tank and fluid properties:
            - 'omega_s' (float): Natural sloshing frequency (rad/s).
            - 'zeta_s' (float): Damping ratio.
            - 'alpha_s' (float): Nonlinear spring coefficient.
            - 'a' (float): Tank radius (m).
            - 'm_s' (float): Sloshing mass (kg).
            - 'r_s_tank' (numpy.ndarray): Offset of sloshing mass from tank pivot [x, y, z] (m).
            - 'r_tank_to_cog' (numpy.ndarray): Offset vector from tank pivot to rocket CoG [x, y, z] (m).

    Returns:
        numpy.ndarray: Sloshing force vector [F_x, F_y, F_z] (N).
        numpy.ndarray: Sloshing moment vector [M_x, M_y, M_z] (Nm).
    """
    # Extract parameters
    omega_s = params['omega_s']
    zeta_s = params['zeta_s']
    alpha_s = params['alpha_s']
    a = params['a']
    m_s = params['m_s']
    r_s_tank = np.array(params['r_s_tank'])
    r_tank_to_cog = np.array(params['r_tank_to_cog'])

    # Magnitude of displacement
    r_s_mag = np.linalg.norm(delta_s)

    # Nonlinear damping matrix
    I = np.eye(2)  # Identity matrix
    damping_matrix = I + (omega_s ** 4 / params['g_bar'] ** 2) * np.outer(delta_s, delta_s)

    # Compute sloshing force
    force_slosh = (
        -2 * omega_s * zeta_s * np.dot(damping_matrix, delta_s_dot)
        - omega_s**2 * delta_s
        - (omega_s**4 / params['g_bar']**2) * r_s_mag**2 * delta_s
        - omega_s**2 * (alpha_s / a**2) * r_s_mag**2 * delta_s
    )

    # Include vehicle dynamics forcing terms
    force_slosh += m_s * (
        omega_dot + np.cross(omega, np.cross(omega, delta_s)) - 2 * np.cross(omega, delta_s_dot)
    )

    # Compute moments around tank pivot
    moment_slosh_tank = np.cross(r_s_tank, force_slosh)

    # Adjust moments to rocket CoG
    moment_slosh_cog = moment_slosh_tank + np.cross(r_tank_to_cog, force_slosh)

    return force_slosh, moment_slosh_cog

# Body reference frame: x through nose, y to right, z through belly