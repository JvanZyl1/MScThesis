import pickle
import numpy as np
from scipy.interpolate import RegularGridInterpolator

# Load dataset
with open('dataset_new_new.pkl', 'rb') as f:
    dataset = pickle.load(f)

H = 2.0
pitch_vals = np.linspace(-np.pi/2, np.pi/2, 20)
roll_vals = np.linspace(-np.pi/2, np.pi/2, 20)
yaw_vals = np.linspace(-np.pi/2, np.pi/2, 20)
fill_level_vals = np.linspace(0, H, 20)

# Convert dataset to numpy array for easier manipulation
dataset = np.array(dataset, dtype=object)

# Create empty arrays to store inertia values for the 4D grid
I_xx_grid = np.zeros((len(yaw_vals), len(fill_level_vals), len(pitch_vals), len(roll_vals)))
I_yy_grid = np.zeros((len(yaw_vals), len(fill_level_vals), len(pitch_vals), len(roll_vals)))
I_zz_grid = np.zeros((len(yaw_vals), len(fill_level_vals), len(pitch_vals), len(roll_vals)))

# Fill the 4D arrays with the dataset values
yaw_indices = range(len(yaw_vals))
fill_indices = range(len(fill_level_vals))
for yaw_index in yaw_indices:
    for fill_index in fill_indices:
        yaw_value_to_plot = yaw_vals[yaw_index]
        fill_level_value_to_plot = fill_level_vals[fill_index]
        yaw_to_plot = np.abs(dataset[:, 2] - yaw_value_to_plot) < 0.1
        fill_level_to_plot = np.abs(dataset[:, 3] - fill_level_value_to_plot) < 0.1
        filtered_data = dataset[yaw_to_plot & fill_level_to_plot]

        # Check if filtered_data is empty
        if len(filtered_data) == 0:
            continue

        # Extract data for filling the grid
        pitch_vals_filtered = filtered_data[:, 0]
        roll_vals_filtered = filtered_data[:, 1]
        inertia_matrices = filtered_data[:, 4]

        # Assuming inertia matrix I is a 3x3 matrix, extract symmetric components
        I_xx = [I[0, 0] for I in inertia_matrices]
        I_yy = [I[1, 1] for I in inertia_matrices]
        I_zz = [I[2, 2] for I in inertia_matrices]

        # Fill the grid with inertia values
        for pitch, roll, xx, yy, zz in zip(pitch_vals_filtered, roll_vals_filtered, I_xx, I_yy, I_zz):
            pitch_index = np.where(pitch_vals == pitch)[0][0]
            roll_index = np.where(roll_vals == roll)[0][0]
            I_xx_grid[yaw_index, fill_index, pitch_index, roll_index] = xx
            I_yy_grid[yaw_index, fill_index, pitch_index, roll_index] = yy
            I_zz_grid[yaw_index, fill_index, pitch_index, roll_index] = zz

# Create interpolators for each component of the inertia matrix
interp_I_xx = RegularGridInterpolator((yaw_vals, fill_level_vals, pitch_vals, roll_vals), I_xx_grid, bounds_error=False, fill_value=None)
interp_I_yy = RegularGridInterpolator((yaw_vals, fill_level_vals, pitch_vals, roll_vals), I_yy_grid, bounds_error=False, fill_value=None)
interp_I_zz = RegularGridInterpolator((yaw_vals, fill_level_vals, pitch_vals, roll_vals), I_zz_grid, bounds_error=False, fill_value=None)

# Example usage of the interpolator
# Input: yaw, fill level, pitch, roll
yaw_input = 0.1
fill_level_input = 1.0
pitch_input = 0.2
roll_input = -0.3

# Interpolating values for I_xx, I_yy, I_zz
I_xx_interp = interp_I_xx((yaw_input, fill_level_input, pitch_input, roll_input))
I_yy_interp = interp_I_yy((yaw_input, fill_level_input, pitch_input, roll_input))
I_zz_interp = interp_I_zz((yaw_input, fill_level_input, pitch_input, roll_input))

# Print the interpolated inertia values
print(f'Interpolated I_xx: {I_xx_interp:.4f}')
print(f'Interpolated I_yy: {I_yy_interp:.4f}')
print(f'Interpolated I_zz: {I_zz_interp:.4f}')
