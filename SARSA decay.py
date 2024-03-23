import gymnasium as gym
import random 
import matplotlib.pyplot as plt
import numpy as np

CONST_WINS_FOLDER = "results/wins/"

class SARSA():
    def __init__(self, alpha, epsilon):
        # Algorithm parameters: step size alpha within (0, 1], small epsilon > 0       
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = 0.9
        
        self.wins = []
        
        # Initialize Q(s, a), 
        self.q_values = {}       
        for hand in range(4, 32):
           for dealer in range(1, 11):
               for ace in [0, 1]:
                   for action in [0, 1]:
                       self.q_values[(hand, dealer, ace, action)] = 0
        
        pass

    def update_wins(self, reward):
        self.wins.append(reward)


    ###############################
    #### Epsilon Greedy Policy ####
    ###############################
    
    def policy(self, state):
        hand = state[0]
        dealer = state[1]
        ace = state[2]
            
        if random.random() < self.epsilon:
            #return random action
            return random.randint(0, 1)
            
        else:
            return self.argmax_a_q(hand, dealer, ace)


    ###############################
    ## Argmax of action a in q ####
    ###############################
    
    def argmax_a_q(self, hand, dealer, ace):
        actions = [0 , 1]
        best_action = None
        best_q_value = float('-inf')
        
        for action in actions:
            q_value = self.q_values[(hand, dealer, ace, action)]
            if q_value > best_q_value:
                best_action = action
                best_q_value = q_value

        return best_action
    
    
    ###############################
    ###### Update q Values ########
    ###############################

    def update_Q(self, state, action, reward, state_next, action_next):
        hand = state[0]
        dealer = state[1]
        ace = state[2]
        
        hand_1 = state_next[0]
        dealer_1 = state_next[1]
        ace_1 = state_next[2]      
        
        q_sa = self.q_values[(hand, dealer, ace, action)]
        q_sa_next = self.q_values[(hand_1, dealer_1, ace_1, action_next)]
        
        self.q_values[(hand, dealer, ace, action)] = q_sa + self.alpha * (reward + (self.gamma * q_sa_next) - q_sa)
    
    
    def decay_epsilon(self, t, decay_rate, n0 = 0.1):
        self.epsilon = n0 * np.exp((-1)*decay_rate * t)

########################################################
#################### Plotting ##########################
########################################################  

def plot_learning_curve(wins):
    avg_wins = [sum(wins[:i+1]) / len(wins[:i+1]) for i in range(len(wins))]
    plt.plot(range(1, len(wins) + 1), avg_wins)
    plt.xlabel('Episodes')
    plt.ylabel('Average Win Rate')
    plt.title('Average Win Rate over Episodes SARSA')
    plt.show()
    
    
def calculate_mean_std(wins):
    mean_wins = np.mean(wins)
    std_wins = np.std(wins)
    print("Average wins: ", mean_wins)
    print("Standard Deviation of wins: ", std_wins)
    
    
def save_to_csv_wins(wins, file_name):
    np.savetxt(CONST_WINS_FOLDER + file_name + ".csv",
        wins,
        delimiter =", ",
        fmt ='% s')    
    
########################################################
########################################################  


def gen_episode(agent, t) -> list:  
    
    done = False
    
    decay_rate = 0.05
    n0 = 0.1
    
    # decay epsilon
    agent.epsilon = n0 * np.exp((-1)*decay_rate * t)
    
    # Initialize S
    state = env.reset()[0]
    
    # Choose A from S using policy derived from Q
    action = agent.policy(state)
    
    rewards = []
    
    while not done:
        # Take action A, observe R, S'
        state_next, reward, terminated, truncated, info = env.step(action)
        
        # Choose A' from S' using policy derived from Q
        action_next = agent.policy(state_next)
        
        # Update Q
        agent.update_Q(state, action, reward, state_next, action_next)
        
        # S = s_next; A = a_next
        state = state_next
        action = action_next
        
        rewards.append(reward)
        
        # until S is terminal        
        done = terminated or truncated 
    
    agent.wins.append(sum(rewards))

if __name__ == '__main__':
    env = gym.make("Blackjack-v1", sab=True)
    
    MAX_EPISODES = 10000
    alpha = 0.05
    epsilon = 0.1
    num_games = 50
    
    for i in range(num_games):
    
        agent = SARSA(alpha, epsilon)
        
        for t in range(1, MAX_EPISODES + 1):
            
            gen_episode(agent, t)
        
        plot_learning_curve(agent.wins)
        calculate_mean_std(agent.wins)
        save_to_csv_wins(agent.wins, "SARSA_DECAY_ep_" + str(MAX_EPISODES) + "_" + str(i))
    
    
    
    
