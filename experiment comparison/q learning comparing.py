"""
Solving Blackjack with Q-Learning
=================================

"""


from __future__ import annotations

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from tqdm import tqdm

import gymnasium as gym

CONST_WINS_FOLDER = "../results/wins/"
# Let's start by creating the blackjack environment.
# Note: We are going to follow the rules from Sutton & Barto.
# Other versions of the game can be found below for you to experiment.

env = gym.make("Blackjack-v1", sab=True)

class BlackjackAgent:
    def __init__(
        self,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        """Initialize a Reinforcement Learning agent with an empty dictionary
        of state-action values (q_values), a learning rate and an epsilon.

        Args:
            learning_rate: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            discount_factor: The discount factor for computing the Q-value
        """
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

    def get_action(self, obs: tuple[int, int, bool]) -> int:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        # with probability epsilon return a random action to explore the environment
        if np.random.random() < self.epsilon:
            return env.action_space.sample()

        # with probability (1 - epsilon) act greedily (exploit)
        else:
            return int(np.argmax(self.q_values[obs]))

    def update(
        self,
        obs: tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, bool],
    ):
        """Updates the Q-value of an action."""
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        temporal_difference = (
            reward + self.discount_factor * future_q_value - self.q_values[obs][action]
        )

        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * temporal_difference
        )
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - epsilon_decay)

# 
# .. code:: py
#
#   # Other possible environment configurations are:
#
#   env = gym.make('Blackjack-v1', natural=True, sab=False)
#   # Whether to give an additional reward for starting with a natural blackjack, i.e. starting with an ace and ten (sum is 21).
#
#   env = gym.make('Blackjack-v1', natural=False, sab=False)
#   # Whether to follow the exact rules outlined in the book by Sutton and Barto. If `sab` is `True`, the keyword argument `natural` will be ignored.
#


# 
# Observing the environment
# ------------------------------
#
# First of all, we call ``env.reset()`` to start an episode. This function
# resets the environment to a starting position and returns an initial
# ``observation``. We usually also set ``done = False``. This variable
# will be useful later to check if a game is terminated (i.e., the player wins or loses).
#

# reset the environment to get the first observation
done = False
observation, info = env.reset()

# observation = (16, 9, False)


# 
# Note that our observation is a 3-tuple consisting of 3 values:
#
# -  The players current sum
# -  Value of the dealers face-up card
# -  Boolean whether the player holds a usable ace (An ace is usable if it
#    counts as 11 without busting)
#


# 
# Executing an action
# ------------------------------
#
# After receiving our first observation, we are only going to use the
# ``env.step(action)`` function to interact with the environment. This
# function takes an action as input and executes it in the environment.
# Because that action changes the state of the environment, it returns
# four useful variables to us. These are:
#
# -  ``next_state``: This is the observation that the agent will receive
#    after taking the action.
# -  ``reward``: This is the reward that the agent will receive after
#    taking the action.
# -  ``terminated``: This is a boolean variable that indicates whether or
#    not the environment has terminated.
# -  ``truncated``: This is a boolean variable that also indicates whether
#    the episode ended by early truncation, i.e., a time limit is reached.
# -  ``info``: This is a dictionary that might contain additional
#    information about the environment.
#
# The ``next_state``, ``reward``,  ``terminated`` and ``truncated`` variables are
# self-explanatory, but the ``info`` variable requires some additional
# explanation. This variable contains a dictionary that might have some
# extra information about the environment, but in the Blackjack-v1
# environment you can ignore it. For example in Atari environments the
# info dictionary has a ``ale.lives`` key that tells us how many lives the
# agent has left. If the agent has 0 lives, then the episode is over.
#
# Note that it is not a good idea to call ``env.render()`` in your training
# loop because rendering slows down training by a lot. Rather try to build
# an extra loop to evaluate and showcase the agent after training.
#

# sample a random action from all valid actions
action = env.action_space.sample()
# action=1

# execute the action in our environment and receive infos from the environment
observation, reward, terminated, truncated, info = env.step(action)

# observation=(24, 10, False)
# reward=-1.0
# terminated=True
# truncated=False
# info={}


# 
# Once ``terminated = True`` or ``truncated=True``, we should stop the
# current episode and begin a new one with ``env.reset()``. If you
# continue executing actions without resetting the environment, it still
# responds but the output won’t be useful for training (it might even be
# harmful if the agent learns on invalid data).
#


# 
# Building an agent
# ------------------------------
#
# Let’s build a ``Q-learning agent`` to solve *Blackjack-v1*! We’ll need
# some functions for picking an action and updating the agents action
# values. To ensure that the agents explores the environment, one possible
# solution is the ``epsilon-greedy`` strategy, where we pick a random
# action with the percentage ``epsilon`` and the greedy action (currently
# valued as the best) ``1 - epsilon``.
#


# 
# To train the agent, we will let the agent play one episode (one complete
# game is called an episode) at a time and then update it’s Q-values after
# each episode. The agent will have to experience a lot of episodes to
# explore the environment sufficiently.
#
# Now we should be ready to build the training loop.
#

# hyperparameters
learning_rate = 0.01
n_episodes = 10_000
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2)  # reduce the exploration over time
final_epsilon = 0.1

agent = BlackjackAgent(
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
)

# 
# Great, let’s train!
#
# Info: The current hyperparameters are set to quickly train a decent agent.
# If you want to converge to the optimal policy, try increasing
# the n_episodes by 10x and lower the learning_rate (e.g. to 0.001).
#


env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=n_episodes)
i = 49
wins = []

for episode in tqdm(range(n_episodes)):
    obs, info = env.reset()
    done = False

    rewards = []

    # play one episode
    while not done:
        action = agent.get_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        rewards.append(reward)
        
        # update the agent
        agent.update(obs, action, reward, terminated, next_obs)

        # update if the environment is done and the current obs
        done = terminated or truncated
        obs = next_obs

    agent.decay_epsilon()
    
    final_reward = sum(rewards)
    wins.append(final_reward)

file_name = "q-learning"
np.savetxt(CONST_WINS_FOLDER + file_name +"_" + str(i) + ".csv", wins, delimiter =", ", fmt ='% s')
