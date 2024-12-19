import mujoco_py as mujoco
import numpy as np
import os

def fuel_depletion(sim):
    # Parameters
    max_fuel = 5.0  # Initial fuel mass
    fuel_rate = 0.1  # Fuel consumption rate per thrust unit
    dt = sim.model.opt.timestep  # Simulation timestep

    # Deplete fuel
    fuel_mass = max(0.0, sim.data.userdata[0] - fuel_rate * sim.data.ctrl[0] * dt)
    sim.data.userdata[0] = fuel_mass  # Store remaining fuel mass

    # Update rocket mass and inertia
    total_mass = 10 + fuel_mass  # Rocket dry mass + fuel mass
    sim.model.body_mass[sim.model.body_name2id("rocket")] = total_mass
    # Update inertia (simplified as a cylinder with variable mass)
    inertia = 0.5 * total_mass * (0.5**2)  # For a cylinder
    sim.model.body_inertia[sim.model.body_name2id("rocket")] = [inertia, inertia, inertia]

def main():
    # Load Model
    xml_path = os.path.join(os.path.dirname(__file__), "rocket.xml")
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"XML file not found: {xml_path}")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    sim = mujoco.MjSim(model, data)

    # Create viewer
    viewer = mujoco.MjViewer(sim)

    # Hook custom function
    sim.set_callback(lambda: fuel_depletion(sim))

    # Run Simulation
    while True:
        sim.step()
        viewer.render()
        print(f"Fuel Mass: {sim.data.userdata[0]:.2f}, Total Mass: {model.body_mass[model.body_name2id('rocket')]:.2f}")

if __name__ == "__main__":
    main()
