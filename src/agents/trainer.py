from src.utils.saving_data import save_agent, save_losses
from tqdm import tqdm
import matplotlib.pyplot as plt

class Trainer:
    '''
    Class to train the MPO agent.
    '''
    def __init__(self, env, agent, num_epochs, batch_size, save_model_interval, model_name, note):
        self.env = env
        self.agent = agent
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.save_model_interval = save_model_interval
        self.model_name = model_name
        self.note = note
        self.critic_losses = []
        self.actor_losses = []

    def train(self):
        '''
        Train the agent and plot results.
        '''
        # Initial phase to fill the replay buffer
        print("Filling replay buffer...")
        while len(self.agent.replay_buffer) < self.batch_size:
            state = self.env.reset()
            done = False
            while not done:
                action = self.env.action_space.sample()  # Take random actions to fill the buffer
                next_state, reward, done, truncated, _ = self.env.step(action)
                done = done or truncated
                self.agent.replay_buffer.add(state, action, reward, next_state, done, td_error=0.0)
                state = next_state

        print("Replay buffer filled. Starting training...")

        # Training phase
        for epoch in tqdm(range(self.num_epochs), desc="Training Progress"):
            # Ensure the replay buffer has enough samples before updating
            if len(self.agent.replay_buffer) >= self.batch_size:
                critic_loss, actor_loss = self.agent.update(self.batch_size)
                self.critic_losses.append(critic_loss)
                self.actor_losses.append(actor_loss)

                print(f"Epoch: {epoch + 1}/{self.num_epochs}, Critic Loss: {critic_loss:.4f}, Actor Loss: {actor_loss:.4f}")

                # Save agent every save_model_interval epochs
                if (epoch + 1) % self.save_model_interval == 0:
                    note = f"{self.note}, Epoch: {epoch + 1}/{self.num_epochs}"
                    save_agent(self.agent, self.model_name, note=note)
                    save_losses(self.critic_losses, self.actor_losses, note=note)
            else:
                print(f"Epoch: {epoch + 1}/{self.num_epochs}, Not enough samples in replay buffer to update.")

        # Plot training curves
        self.plot_training_curves()

    def plot_training_curves(self):
        '''
        Plot the critic and actor losses over epochs.
        '''
        plt.figure(figsize=(12, 6))

        # Critic Loss Plot
        plt.subplot(1, 2, 1)
        if self.critic_losses:
            plt.plot(range(1, len(self.critic_losses) + 1), self.critic_losses, label='Critic Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Critic Loss over Training')
        plt.legend()

        # Actor Loss Plot
        plt.subplot(1, 2, 2)
        if self.actor_losses:
            plt.plot(range(1, len(self.actor_losses) + 1), self.actor_losses, label='Actor Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Actor Loss over Training')
        plt.legend()

        plt.tight_layout()
        plt.show()