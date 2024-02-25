import gymnasium as gym
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
STARTING_VALUES = 0.5

class Blackjack_Agent():
    def __init__(self, epsilon, time_value):
        self.epsilon = epsilon
        self.time_value = time_value #should be 1 for blackjack, because it is episodic
        self.episode_count = 0
        
        self.reward_record = []

        # this makes it not-generic
        # 3-d array - [our_hand][dealer_hand][usable_ace]
        self.state_values = [[[STARTING_VALUES] * 2] * 11] * 32
        self.state_times_visited = [[[0] * 2] * 11] * 32

        # 4-d array - [our_hand][dealer_hand][usable_ace][stick, hit]
        self.state_action_values = [[[[STARTING_VALUES] * 2] * 2] * 11] * 32
        self.state_action_times_visited = [[[[0] * 2] * 2] * 11] * 32
        

        self.state = (0,0,0)

    def get_current_state_values(self):
        return self.state_values[self.state[0]][self.state[1]][self.state[2]]
    
    def generate_env(self):
        self.env = gym.make("Blackjack-v1", sab=True, render_mode="human")
    
    def update_state(self, new_state):
        self.state = new_state
    
    def update_state_values(self, previous_state, reward): #TD update
        # TD-learning state-value update function
        self.state_times_visited[previous_state[0]][previous_state[1]][previous_state[2]] += 1

        self.state_values[previous_state[0]][previous_state[1]][previous_state[2]] = \
            self.state_values[previous_state[0]][previous_state[1]][previous_state[2]] + \
                (1 / self.state_times_visited[previous_state[0]][previous_state[1]][previous_state[2]]) * (reward + self.time_value * (self.get_current_state_values()) - \
                                      self.state_values[previous_state[0]][previous_state[1]][previous_state[2]])
        

    def evaluate_policy(self):
        pass

    def update_state_action(self, previous_state, action, reward):
        self.state_action_times_visited[previous_state[0]][previous_state[1]][previous_state[2]][action] += 1
        #For adjusting alpha

        self.state_action_values[previous_state[0]][previous_state[1]][previous_state[2]][action] = \
            self.state_action_values[previous_state[0]][previous_state[1]][previous_state[2]][action] + \
                (1 / self.state_action_times_visited[previous_state[0]][previous_state[1]][previous_state[2]][action]) * \
                    (reward + self.time_value * (self.get_current_state_values()) - \
                                      self.state_action_values[previous_state[0]][previous_state[1]][previous_state[2]][action])

    def update_policy(self, previous_state, action, reward):
        #self.policy[self.state]
        #value for a state-action pair = previous value for that state-action pair + [learning_rate](Reward + time_value(value of the best possible action you could take given the next state) - previous_value)
        
        self.update_state_values(previous_state, reward)
        self.update_state_action(previous_state, action, reward)
        
        #epsilon-greedy

    def get_state_action_values(self, state, action):
        return self.state_action_values[state[0]][state[1]][state[2]][action]
    
    def pick_action(self):
        if random.random() >= self.epsilon:
            if self.get_state_action_values(self.state, 1) > self.get_state_action_values(self.state, 0):
                return 1 # Hit
            else:
                return 0 # Stick
            
        else:
            return random.choice([0, 1]) # Picks randomly
        
    def decay_epsilon(self):
        self.epsilon = 1/self.episode_count

    def get_reward_record(self):
        return self.reward_record
    
    def add_episode_to_record(self, episode):
        self.reward_record.append(episode)

    def close_env(self):
        self.env.close()

def run_episode(agent):
    agent.state = agent.env.reset()[0]
    agent.episode_count += 1
    agent.decay_epsilon()
    episode_reward = 0 #this methodology dependent on time_value = 1. modifications needed to make it adaptable
    done = False

    while not done:
        previous_state = agent.state

        action = agent.pick_action()

        state_n_1, reward_n, done, truncated, info = agent.env.step(action)
        agent.update_state(state_n_1)

        agent.update_policy(previous_state, action, reward_n)

        episode_reward += reward_n
    
    return episode_reward

def run_learning_loop(agent, num_episodes):
    while agent.episode_count < num_episodes:
        agent.add_episode_to_record(run_episode(agent))
    
    agent.close_env()

    plot_values(agent.get_reward_record())



        

def plot_values(values):
    
    
    
    plt.scatter(range(len(values)), values, s=3)
    plt.xlabel("Experiment Number")
    plt.ylabel("Reward Received")
    plt.title("Reward Received Over Time")

    plt.show()




        



if __name__ == '__main__':
    MyAgent = Blackjack_Agent(0.1, 1)
    MyAgent.generate_env()

    run_learning_loop(MyAgent, 50)
