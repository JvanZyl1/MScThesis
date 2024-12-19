import gymnasium as gym
from src.agents.mpo import MPOLearner
from configs.agents_parameters import config
from src.agents.trainer import Trainer

# RUN: python -m notebooks.MPO_trainer

def main():
    # Initialize the MuJoCo environment (Inverted Pendulum)
    env = gym.make("LunarLanderContinuous-v3")

    # Extract state and action dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Configure the MPO learner
    agent = MPOLearner(
        state_dim=state_dim,
        action_dim=action_dim,
        config=config,
    )

    # Initialize the Trainer
    num_epochs = 10000
    batch_size = 10
    trainer = Trainer(
        env = env,
        agent=agent,
        num_epochs=num_epochs,
        batch_size=batch_size,
        save_model_interval= num_epochs+1,#config["save_model_interval"],
        model_name="mpo_inverted_pendulum",
        note="Training MPO on Inverted Pendulum using doubledistributionalcritics_Nstep_Target_PER"
    )

    # Train the agent
    trainer.train()

    # Close the environment
    env.close()

if __name__ == "__main__":
    main()
