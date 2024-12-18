import sys
import os

# Initialize the environment by importing the notebooks package
import notebooks

# Now import and run the trainer
from notebooks.MPO_trainer import main  # Ensure MPO_trainer.py has a main function

if __name__ == "__main__":
    main()