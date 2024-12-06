import numpy as np
from collections import deque
import random

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        states = np.stack(states)
        next_states = np.stack(next_states)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)
    
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        '''
        params:
        capacity: max number of samples to store in the buffer [int]
        alpha: determines how much prioritization is used [float]
            -> alpha=0 is uniform sampling
            -> alpha=1 is full prioritization

        PER paper: https://arxiv.org/abs/1511.05952
            -> " measured by the magnitude of their temporal-difference (TD) error"
        '''

        self.capacity = capacity
        self.buffer = []                                            # list of experiences
        self.priorities = np.zeros((capacity,), dtype=np.float32)   # list of priorities
        self.pos = 0                                                # position in buffer
        self.alpha = alpha                                          # alpha value for prioritization

    def add(self, state, action, reward, next_state, done, td_error):
        if len(self.buffer) == self.capacity: # Replace the oldest experience with the new one
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        else: # Add a new experience
            self.buffer.append((state, action, reward, next_state, done))

        # Calculate the priority as the TD error + small epsilon
        priority = td_error + 1e-5
        # Add the priority to the priority list
        self.priorities[self.pos] = priority
        # Increment the position
        self.pos = (self.pos + 1) % self.capacity

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

    def sample(self, batch_size, beta=0.4):
        '''
        params:
        batch_size: number of samples to return [int]
        beta: determines how much importance sampling is used [float]
            -> beta=0 is no importance sampling
            -> beta=1 is full importance sampling
            ~ importance: "weights are the inverse of the probability of sampling that transition"
        
        "Furthermore, Hinton (2007) introduced a form of non-uniform sampling based on error,
        with an importance sampling correction, which led
        to a 3x speed-up on MNIST digit classification." https://arxiv.org/abs/1511.05952
        '''
        
        # Calculate the sampling probability for each sample
        if len(self.buffer) == self.capacity: # If the buffer is full, use all the priorities
            priorities = self.priorities
        else: # If the buffer is not full, use only the priorities up to the current position
            priorities = self.priorities[:self.pos]
        
        # Calculate the probability of each sample being chosen: priority^alpha / sum(priority^alpha)
        probabilities = (priorities ** self.alpha) / (priorities ** self.alpha).sum() # Eq.1 in PER paper.
        
        buffer_size = len(self.buffer)
        # Choose samples based on the probabilities
        indices = np.random.choice(buffer_size,
                                   batch_size,
                                   p=probabilities)
        samples = []
        for idx in indices:
            samples.append(self.buffer[idx])
        
        # Standard Importance Sampling Weight Equation : (1/N * 1/P(i)) ^ beta
        weights = (probabilities[indices] * len(self.buffer)) ** (-beta) # Eq in Sec3.4 in PER paper.
        #Normalized Importance Sampling Weights
        # "For stability reasons, we always normalize weights by 1/ maxi wi so
        # that they only scale the update downwards.""
        weights /= weights.max()

        # Return the samples, indices, and weights
        states, actions, rewards, next_states, dones = zip(*samples)
        states = np.stack(states)
        next_states = np.stack(next_states)
        return_tuple = (states, actions, rewards, next_states, dones, indices, weights)
        return return_tuple

    def __len__(self):
        return len(self.buffer)