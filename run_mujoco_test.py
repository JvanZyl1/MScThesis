import sys
import os

# Initialize the environment by importing the notebooks package
import tests

from tests.mujoco_casestudy.rocket_mujo_co_test import main

# Ensure the working directory is set to the script's location
os.chdir(os.path.dirname(__file__))

# Run the main function
main()