import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

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

# Select a yaw and fill level for visualization (allowing for a small range)
yaw_value_to_plot = yaw_vals[4]  # Taking the 5th yaw angle
fill_level_value_to_plot = fill_level_vals[1]  # Taking the 2nd fill level
yaw_to_plot = np.abs(dataset[:, 2] - yaw_value_to_plot) < 0.1  # Taking the 5th yaw angle value with some tolerance
fill_level_to_plot = np.abs(dataset[:, 3] - fill_level_value_to_plot) < 0.1  # Taking the 2nd fill level value with some tolerance
filtered_data = dataset[yaw_to_plot & fill_level_to_plot]

# Check if filtered_data is empty
if len(filtered_data) == 0:
    print("No data points found for the specified yaw and fill level conditions.")
else:
    # Extract data for plotting
    pitch_vals = filtered_data[:, 0]
    roll_vals = filtered_data[:, 1]
    inertia_matrices = filtered_data[:, 4]

    # Assuming inertia matrix I is a 3x3 matrix, extract symmetric components
    I_xx = [I[0, 0] for I in inertia_matrices]
    I_yy = [I[1, 1] for I in inertia_matrices]
    I_zz = [I[2, 2] for I in inertia_matrices]

    # Create a grid for contour plotting
    grid_x, grid_y = np.meshgrid(np.linspace(min(roll_vals), max(roll_vals), 100),
                                 np.linspace(min(pitch_vals), max(pitch_vals), 100))

    # Interpolate I_xx, I_yy, I_zz values onto the grid
    grid_I_xx = griddata((roll_vals, pitch_vals), I_xx, (grid_x, grid_y), method='cubic')
    grid_I_yy = griddata((roll_vals, pitch_vals), I_yy, (grid_x, grid_y), method='cubic')
    grid_I_zz = griddata((roll_vals, pitch_vals), I_zz, (grid_x, grid_y), method='cubic')

    # Plotting the symmetric entries as contour plots
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Contour plot for I_xx
    cs1 = axs[0].contourf(grid_x, grid_y, grid_I_xx, levels=20, cmap='viridis')
    axs[0].set_xlabel('Roll (rad)')
    axs[0].set_ylabel('Pitch (rad)')
    axs[0].set_title('I_xx Contour')
    fig.colorbar(cs1, ax=axs[0])

    # Contour plot for I_yy
    cs2 = axs[1].contourf(grid_x, grid_y, grid_I_yy, levels=20, cmap='plasma')
    axs[1].set_xlabel('Roll (rad)')
    axs[1].set_ylabel('Pitch (rad)')
    axs[1].set_title('I_yy Contour')
    fig.colorbar(cs2, ax=axs[1])

    # Contour plot for I_zz
    cs3 = axs[2].contourf(grid_x, grid_y, grid_I_zz, levels=20, cmap='inferno')
    axs[2].set_xlabel('Roll (rad)')
    axs[2].set_ylabel('Pitch (rad)')
    axs[2].set_title('I_zz Contour')
    fig.colorbar(cs3, ax=axs[2])

        # Set the overall title with yaw angle and fill level
    yaw_value = yaw_value_to_plot  # Yaw value used for filtering
    fill_value = fill_level_value_to_plot  # Fill level used for filtering
    fig.suptitle(f'Contour Plots for Yaw Angle ~ {yaw_value:.2f} rad, Fill Level ~ {fill_value:.2f} m', fontsize=16)

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Adjust to fit the suptitle
    plt.show()
