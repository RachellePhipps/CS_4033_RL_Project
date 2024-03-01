import gymnasium as gym
import random
import numpy as np
from numpy.polynomial.polynomial import polyfit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
STARTING_VALUES = 0.5
FIRST_EPSILON_DECAY = 1000
CHUNK_SIZE = 10

class Blackjack_Agent():
    def __init__(self, epsilon, time_value):
        self.epsilon = epsilon
        self.time_value = time_value #should be 1 for blackjack, because it is episodic
        self.episode_count = 0
        
        self.reward_record = []
        self.chunked_record = []
        self.current_chunk = 0

        # this makes it not-generic
        # 3-d array - [our_hand][dealer_hand][usable_ace]

        ### These variables are unnecessary in q-learning
        #self.state_values = [[[STARTING_VALUES] * 2] * 11] * 32
        #self.state_times_visited = [[[0] * 2] * 11] * 32

        # 4-d array - [our_hand][dealer_hand][usable_ace][stick, hit]
        self.state_action_values = [[[[STARTING_VALUES] * 2] * 2] * 11] * 32
        self.state_action_times_visited = [[[[0] * 2] * 2] * 11] * 32
        
        self.state = (0,0,0)

    def get_current_state_values(self):
        return self.state_values[self.state[0]][self.state[1]][self.state[2]]
    
    def generate_env(self):
        self.env = gym.make("Blackjack-v1", sab=True)
    
    def update_state(self, new_state):
        self.state = new_state
    
    # This function is unnecessary in q-learning
    '''def update_state_values(self, previous_state, reward): #TD update
        # TD-learning state-value update function
        self.state_times_visited[previous_state[0]][previous_state[1]][previous_state[2]] += 1

        self.state_values[previous_state[0]][previous_state[1]][previous_state[2]] = \
            self.state_values[previous_state[0]][previous_state[1]][previous_state[2]] + \
                (1 / self.state_times_visited[previous_state[0]][previous_state[1]][previous_state[2]]) * (reward + self.time_value * (self.get_current_state_values()) - \
                                      self.state_values[previous_state[0]][previous_state[1]][previous_state[2]])
        '''


    def update_state_action(self, previous_state, action_taken, reward):
        self.state_action_times_visited[previous_state[0]][previous_state[1]][previous_state[2]][action_taken] += 1
        #For adjusting alpha

        # Q(Last state, action taken) = Q(Last state, action taken) + (1/times_visited)(Measured_reward+gamma(Best state-action value in the next state)-Q(last_state, action_taken))

        self.state_action_values[previous_state[0]][previous_state[1]][previous_state[2]][action_taken] = \
            self.get_state_action_values(previous_state, action_taken) + \
                (1 / self.state_action_times_visited[previous_state[0]][previous_state[1]][previous_state[2]][action_taken]) * \
                    (reward + self.time_value * (max(self.get_state_action_values(self.state, 0), self.get_state_action_values(self.state, 1))) - \
                                      self.get_state_action_values(previous_state, action_taken))
        #print(previous - self.get_state_action_values(previous_state, action))

    def update_policy(self, previous_state, action, reward):
        #self.policy[self.state]
        #value for a state-action pair = previous value for that state-action pair + [learning_rate](Reward + time_value(value of the best possible action you could take given the next state) - previous_value)
        
        # self.update_state_values(previous_state, reward)
        self.update_state_action(previous_state, action, reward)
        
        #epsilon-greedy

    def get_state_action_values(self, state, action):
        return self.state_action_values[state[0]][state[1]][state[2]][action]
    
    def pick_action(self):
        if random.random() > self.epsilon:
            if self.get_state_action_values(self.state, 1) > self.get_state_action_values(self.state, 0):
                return 1 # Hit
            else:
                return 0 # Stick
            
        else:
            return random.choice([0, 1]) # Picks randomly
        
    def decay_epsilon(self):
        if self.episode_count > FIRST_EPSILON_DECAY:
            self.epsilon = 1/self.episode_count

    def get_reward_record(self):
        return self.reward_record
    
    def add_episode_to_record(self, episode):
        # Adds the reward for an episode measured to our records

        self.reward_record.append(episode)
        self.current_chunk += episode
        if self.episode_count % CHUNK_SIZE == 0:
            self.chunked_record.append(self.current_chunk)
            self.current_chunk = 0
        else:
            self.current_chunk += episode

    def close_env(self):
        self.env.close()

def run_episode(agent):
    agent.state = agent.env.reset()[0]
    agent.episode_count += 1
    #agent.decay_epsilon()
    episode_reward = 0 
    done = False
    step = 0

    while not done:
        previous_state = agent.state

        action = agent.pick_action()

        state_n_1, reward_n, done, truncated, info = agent.env.step(action)
        agent.update_state(state_n_1)

        agent.update_policy(previous_state, action, reward_n)

        episode_reward += reward_n * (agent.time_value ** step)

        step += 1
    
    return episode_reward

def run_learning_loop(agent, num_episodes):
    while agent.episode_count < num_episodes:
        agent.add_episode_to_record(run_episode(agent))
    
    agent.close_env()

    #plot_best_action(agent.state_action_values, 0)
    #plot_best_action(agent.state_action_values, 1)
    #plot_values(agent.state_values, 0)
    #plot_values(agent.state_values, 1)
    plot_reward(agent.get_reward_record())


def plot_best_action(state_action, usable_ace):
    x_points = []
    y_points = []
    z_points = []
    for x in range(len(state_action)):
        for y in range(len(state_action[x])):
                if state_action[x][y][usable_ace][1] > state_action[x][y][usable_ace][0]:
                    x_points.append(x)
                    y_points.append(y)
                    z_points.append(1)
                else:
                    x_points.append(x)
                    y_points.append(y)
                    z_points.append(0)

    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d')
    surf=ax.plot_trisurf(y_points, x_points, z_points, cmap=plt.cm.inferno, linewidth=0.2)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


def plot_values(values, usable_aces):
   # Completely stolen from Rachelle 

   # get x, y, and z from values
    x = []
    y = []
    z = []

    #non_usable_aces
    for player_hand in range(len(values)):
        for dealer_hand in range(len(values[player_hand])):
            x.append(player_hand)
            y.append(dealer_hand)
            z.append(values[player_hand][dealer_hand][usable_aces])

   # print(x)
   # print(y)
   # print(z)
    
    # to Add a color bar which maps values to colors.
    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d')
    surf=ax.plot_trisurf(y, x, z, cmap=plt.cm.inferno, linewidth=0.2)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
    #print([[[i for i in j if i > 0] for j in k] for k in values])

def plot_reward(rewards):
    
    x = range(len(rewards))

    #plt.scatter(range(len(values)), values, s=3)
    b, m = polyfit(x, rewards, 1)

    plt.plot(x, rewards, '.')
    plt.plot(x, b + m * x, '-')

    plt.xlabel("Experiment Number")
    plt.ylabel("Reward Received")
    plt.title("Reward Received Over Time")

    plt.show()




        



if __name__ == '__main__':
    MyAgent = Blackjack_Agent(0.1, 1)
    MyAgent.generate_env()

    run_learning_loop(MyAgent, 1000)
