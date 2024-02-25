import gymnasium as gym
import random

class Blackjack_Agent():
    def __init__(self, epsilon, learning_factor, time_value):
        self.epsilon = epsilon
        self.time_value = time_value #should be 1 for blackjack, because it is episodic
        self.learning_factor = learning_factor
        self.probability_matrix[21][10][2][21][10][2]

        # this makes it not-generic
        self.state_values[21][10][2] 
        # 3-d array - [our_hand][dealer_hand][usable_ace]
        
        self.policy[21][11][2] # assumes 2 options - true or false

        self.state[3] = {0,0,0}

    def get_state(self):
        return self.state_values[self.state[0]][self.state[1]][self.state[2]]
    
    def update_state(self, new_state):
        self.state = new_state
    
    def update_state_values(self, previous_state, reward): #TD update
        # TD-learning state-value update function
        self.state_values[previous_state[0]][previous_state[1]][previous_state[2]] = \
            self.state_values[previous_state[0]][previous_state[1]][previous_state[2]] + \
                self.learning_rate * (reward + self.time_value * (self.get_state()) - \
                                      self.state_values[previous_state[0]][previous_state[1]][previous_state[2]])
        

    def evaluate_policy(self):
        pass

    def update_state_action(self, previous_state, action, reward):
        pass

    def update_policy(self, previous_state, action, reward):
        #self.policy[self.state]
        #value for a state-action pair = previous value for that state-action pair + [learning_rate](Reward + time_value(value of the best possible action you could take given the next state) - previous_value)
        
        self.update_state_action(previous_state, action, reward)
        
        #epsilon-greedy

    
    def pick_action(self):
        if random.random >= self.epsilon:
            if random.random() < self.policy[self.state[0]][self.state[1]][self.state[2]]:
                return 1 # Hit
            else:
                return 0 # Stick
            
        else:
            return random.randrange(0, 1, 1) # Picks randomly
        
    
    def decay_epsilon(self):
        pass

def play_round():
    state = env.reset()
    done = False
    
    while not done:
        action = random.choice([0,1])
        observation, reward, done, truncated, info = env.step(action)
        env.render() #TODO: REMOVE THIS!

    return reward



if __name__ == '__main__':
    env = gym.make("Blackjack-v1", sab=True, render_mode="human")
    
    print("Round 1 reward: ", play_round())
    env.close()
