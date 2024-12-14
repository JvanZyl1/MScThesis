from collections import deque
import jax.numpy as jnp
from typing import Tuple
import jax
jax.clear_backends()

class ReplayBuffer:
    '''
    This class implements a uniform replay buffer.
    params:
    capacity: capacity of the replay buffer [int]
    n_step: n-step return [int]
    gamma: discount factor [float]

    returns:
    states: states sampled from the replay buffer [jnp.ndarray]
    actions: actions sampled from the replay buffer [jnp.ndarray]
    rewards: rewards sampled from the replay buffer [jnp.ndarray]
    next_states: next states sampled from the replay buffer [jnp.ndarray]
    dones: dones sampled from the replay buffer [jnp.ndarray]
    '''
    def __init__(self,
                    capacity: int,
                    n_step: int = 1,
                    gamma: float = 0.99):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.n_step_buffer = deque(maxlen=n_step)
        self.n_step = n_step
        self.gamma = gamma

    def add(self,
            state: jnp.ndarray,
            action: jnp.ndarray,
            reward: float,
            next_state: jnp.ndarray,
            done: bool):
        '''
        This function adds a transition to the replay buffer.
        Also, calculates the N-step return if activated.
        params:
        state: state [jnp.ndarray]
        action: action [jnp.ndarray]
        reward: reward [float]
        next_state: next state [jnp.ndarray]
        done: done flag [bool]
        '''
        if self.n_step > 1:
            self.n_step_buffer.append((state, action, reward, next_state, done))

            if len(self.n_step_buffer) < self.n_step:
                return

            reward, next_state, done = self.n_step_cumulative_reward()
            state, action, _, _, _ = self.n_step_buffer[0]

        self.buffer.append((state, action, reward, next_state, done))

    def n_step_cumulative_reward(self):
        '''
        This function calculates the N-step cumulative reward and final state.
        returns:
        reward: N-step cumulative reward [float]
        next_state: next state [jnp.ndarray]
        done: done flag [bool]
        '''
        reward, next_state, done = self.n_step_buffer[-1][-3:]
        for transition in reversed(list(self.n_step_buffer)[:-1]):
            r, n_s, d = transition[-3:]
            reward = r + self.gamma * reward * (1 - d)
            next_state, done = (n_s, d) if d else (next_state, done)
        return reward, next_state, done

    def sample(self,
                batch_size: int,
                rng_key: jax.random.PRNGKey) -> Tuple[jnp.ndarray]:
        '''
        This function samples a batch from the replay buffer.
        params:
        batch_size: size of the batch [int]
        rng_key: key for random number generation [jax.random.PRNGKey]

        returns:
        states: states sampled from the replay buffer [jnp.ndarray]
        actions: actions sampled from the replay buffer [jnp.ndarray]
        rewards: rewards sampled from the replay buffer [jnp.ndarray]
        next_states: next states sampled from the replay buffer [jnp.ndarray]
        dones: dones sampled from the replay buffer [jnp.ndarray]
        '''
        indices = jax.random.choice(rng_key, len(self.buffer), shape=(batch_size,), replace=False)
        samples = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*samples)
        return (
            jnp.stack(states),
            jnp.stack(actions),
            jnp.stack(rewards),
            jnp.stack(next_states),
            jnp.stack(dones),
        )

    def __len__(self) -> int:
        return len(self.buffer)


import jax
import jax.numpy as jnp
from collections import deque
from typing import Tuple

class PrioritizedReplayBuffer:
    '''
    This class implements a prioritized replay buffer.
    params:
    capacity: capacity of the replay buffer [int]
    n_step: n-step return [int]
    gamma: discount factor [float]
    alpha: alpha value for prioritized replay buffer [float]
    beta: initial beta value for importance sampling [float]
    beta_decay: decay rate for beta value [float]

    returns:
    states: states sampled from the replay buffer [jnp.ndarray]
    actions: actions sampled from the replay buffer [jnp.ndarray]
    rewards: rewards sampled from the replay buffer [jnp.ndarray]
    next_states: next states sampled from the replay buffer [jnp.ndarray]
    dones: dones sampled from the replay buffer [jnp.ndarray]
    indices: indices of the samples [list]
    weights: importance weights of the samples [jnp.ndarray]
    '''
    def __init__(self,
                 capacity: int,
                 n_step: int = 1,
                 gamma: float = 0.99,
                 alpha: float = 0.6,
                 beta: float = 0.4,
                 beta_decay: float = 0.001):
        self.capacity = capacity
        self.buffer = []
        self.priorities = jnp.zeros((capacity,), dtype=jnp.float32)
        self.n_step_buffer = deque(maxlen=n_step) if n_step > 1 else None
        self.pos = 0
        self.alpha = alpha
        self.beta = beta
        self.beta_decay = beta_decay
        self.n_step = n_step
        self.gamma = gamma

    def add(self,
            state: jnp.ndarray,
            action: jnp.ndarray,
            reward: float,
            next_state: jnp.ndarray,
            done: bool,
            td_error: float):
        '''
        This function adds a transition to the replay buffer.
        Also, calculates the N-step return if activated.
        Also, calculates the priority of the transition.
        params:
        state: state [jnp.ndarray]
        action: action [jnp.ndarray]
        reward: reward [float]
        next_state: next state [jnp.ndarray]
        done: done flag [bool]
        td_error: TD error [float]
        '''
        if self.n_step > 1:
            self.n_step_buffer.append((state, action, reward, next_state, done))

            if len(self.n_step_buffer) < self.n_step:
                return

            reward, next_state, done = self._get_n_step_info()
            state, action, _, _, _ = self.n_step_buffer[0]

        if len(self.buffer) == self.capacity:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        else:
            self.buffer.append((state, action, reward, next_state, done))

        priority = td_error + 1e-5
        self.priorities = self.priorities.at[self.pos].set(priority)
        self.pos = (self.pos + 1) % self.capacity

    def _get_n_step_info(self) -> Tuple[float, jnp.ndarray, bool]:
        '''
        This function calculates the N-step cumulative reward and final state.
        returns:
        reward: N-step cumulative reward [float]
        next_state: next state [jnp.ndarray]
        done: done flag [bool]
        '''
        reward, next_state, done = self.n_step_buffer[-1][-3:]
        for transition in reversed(list(self.n_step_buffer)[:-1]):
            r, n_s, d = transition[-3:]
            reward = r + self.gamma * reward * (1 - d)
            next_state, done = (n_s, d) if d else (next_state, done)
        return reward, next_state, done

    def update_priorities(self,
                          indices: list,
                          priorities: jnp.ndarray):
        '''
        This function updates the priorities of the samples.
        params:
        indices: indices of the samples [list]
        priorities: priorities of the samples [list]
        '''
        self.priorities = self.priorities.at[indices].set(priorities)

    def sample(self,
               batch_size: int,
               rng_key: jax.random.PRNGKey) -> Tuple[jnp.ndarray]:
        '''
        This function samples a batch from the replay buffer.
        params:
        batch_size: size of the batch [int]
        rng_key: key for random number generation [jax.random.PRNGKey]

        returns:
        states: states sampled from the replay buffer [jnp.ndarray]
        actions: actions sampled from the replay buffer [jnp.ndarray]
        rewards: rewards sampled from the replay buffer [jnp.ndarray]
        next_states: next states sampled from the replay buffer [jnp.ndarray]
        dones: dones sampled from the replay buffer [jnp.ndarray]
        indices: indices of the samples [list]
        weights: importance weights of the samples [jnp.ndarray]

        notes:
        The importance weights are calculated as:
        w = (1/N * 1/P(i)) ** beta

        where:
        N is the size of the replay buffer
        P(i) is the probability of the sample
        beta is the beta value

        The probability of the sample is calculated as:
        P(i) = (p(i) ** alpha) / sum(p(i) ** alpha)
        '''
        if len(self.buffer) < batch_size:
            raise ValueError("Not enough samples in the buffer to sample a batch")

        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[: len(self.buffer)]

        probabilities = (priorities ** self.alpha) / jnp.sum(priorities ** self.alpha)
        indices = jax.random.choice(rng_key, len(self.buffer), shape=(batch_size,), p=probabilities)
        samples = [self.buffer[idx] for idx in indices]

        weights = (probabilities[indices] * len(self.buffer)) ** (-self.beta)
        weights /= jnp.max(weights)

        # Update beta with decay
        self.beta = min(1.0, self.beta + self.beta_decay)

        states, actions, rewards, next_states, dones = map(jnp.array, zip(*samples))
        return (
            jnp.stack(states),
            jnp.stack(actions),
            jnp.stack(rewards),
            jnp.stack(next_states),
            jnp.stack(dones),
            indices,
            weights,
        )

    def __len__(self) -> int:
        return len(self.buffer)