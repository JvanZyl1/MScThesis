import pickle
import numpy as np
from scipy.integrate import tplquad
from tqdm import tqdm

def inertia_matrix(pitch, roll, yaw, fill_fraction, tank_radius, tank_length, fluid_density):
    """
    Computes the inertia matrix of a partially filled cylindrical tank with an inclined free surface,
    accounting for pitch, roll, and yaw angles.

    Parameters:
        pitch (float): Pitch angle of the fluid free surface (radians).
        roll (float): Roll angle of the fluid free surface (radians).
        yaw (float): Yaw angle of the fluid free surface (radians).
        fill_fraction (float): Fill fraction of the tank (0 to 1).
        tank_radius (float): Radius of the cylindrical tank (m).
        tank_length (float): Length of the cylindrical tank (m).
        fluid_density (float): Density of the fluid (kg/m^3).

    Returns:
        numpy.ndarray: 3x3 inertia matrix of the fluid.
    """
    # Set z0 based on the fill fraction
    z0 = -tank_radius + 2 * tank_radius * fill_fraction  # Assuming z0 is the height at the center corresponding to the fill
    
    # Rotation matrix to account for pitch, roll, and yaw
    R = np.array([
        [np.cos(yaw) * np.cos(pitch), np.cos(yaw) * np.sin(pitch) * np.sin(roll) - np.sin(yaw) * np.cos(roll), np.cos(yaw) * np.sin(pitch) * np.cos(roll) + np.sin(yaw) * np.sin(roll)],
        [np.sin(yaw) * np.cos(pitch), np.sin(yaw) * np.sin(pitch) * np.sin(roll) + np.cos(yaw) * np.cos(roll), np.sin(yaw) * np.sin(pitch) * np.cos(roll) - np.cos(yaw) * np.sin(roll)],
        [-np.sin(pitch), np.cos(pitch) * np.sin(roll), np.cos(pitch) * np.cos(roll)]
    ])

    # Integration limits
    def z_min(x):
        return -tank_radius

    def z_max(x):
        return min(tank_radius, z0 + np.tan(pitch) * x)

    def y_min(x, z):
        return -np.sqrt(tank_radius**2 - z**2)

    def y_max(x, z):
        return np.sqrt(tank_radius**2 - z**2)

    # Mass element function
    def mass_element(y, z, x):
        return fluid_density

    # Compute the total mass of the fluid
    mass, _ = tplquad(mass_element, -tank_length / 2, tank_length / 2, z_min, z_max, y_min, y_max)

    # Define integrand functions for moments of inertia
    def I_xx_integrand(y, z, x):
        return fluid_density * (y**2 + z**2)

    def I_yy_integrand(y, z, x):
        return fluid_density * (x**2 + z**2)

    def I_zz_integrand(y, z, x):
        return fluid_density * (x**2 + y**2)

    def I_xy_integrand(y, z, x):
        return -fluid_density * x * y

    def I_xz_integrand(y, z, x):
        return -fluid_density * x * z

    def I_yz_integrand(y, z, x):
        return -fluid_density * y * z

    # Perform the triple integration for each inertia matrix element
    I_xx, _ = tplquad(I_xx_integrand, -tank_length / 2, tank_length / 2, z_min, z_max, y_min, y_max)
    I_yy, _ = tplquad(I_yy_integrand, -tank_length / 2, tank_length / 2, z_min, z_max, y_min, y_max)
    I_zz, _ = tplquad(I_zz_integrand, -tank_length / 2, tank_length / 2, z_min, z_max, y_min, y_max)
    I_xy, _ = tplquad(I_xy_integrand, -tank_length / 2, tank_length / 2, z_min, z_max, y_min, y_max)
    I_xz, _ = tplquad(I_xz_integrand, -tank_length / 2, tank_length / 2, z_min, z_max, y_min, y_max)
    I_yz, _ = tplquad(I_yz_integrand, -tank_length / 2, tank_length / 2, z_min, z_max, y_min, y_max)

    # Construct the inertia matrix
    inertia_matrix = np.array([
        [I_xx, I_xy, I_xz],
        [I_xy, I_yy, I_yz],
        [I_xz, I_yz, I_zz]
    ])

    return inertia_matrix

# Example usage to generate unseen parameter data
pitch_values = np.linspace(0, np.pi / 6, 10)  # 10 values from 0 to 30 degrees
roll_values = np.linspace(0, np.pi / 6, 10)  # 10 values from 0 to 30 degrees
yaw_values = np.linspace(0, np.pi / 6, 10)  # 10 values from 0 to 30 degrees
fill_fraction_values = np.linspace(0.1, 0.9, 10)  # 10 values from 10% to 90% filled

data = []
for pitch in tqdm(pitch_values, desc="Pitch"):
    for roll in tqdm(roll_values, desc="Roll", leave=False):
        for yaw in tqdm(yaw_values, desc="Yaw", leave=False):
            for fill_fraction in tqdm(fill_fraction_values, desc="Fill Fraction", leave=False):
                inertia = inertia_matrix(pitch, roll, yaw, fill_fraction, tank_radius=1.0, tank_length=5.0, fluid_density=1000.0)
                data.append((pitch, roll, yaw, fill_fraction, inertia))
    # Save the dataset to a file
    with open('dataset.pkl', 'wb') as f:
        pickle.dump(data, f)