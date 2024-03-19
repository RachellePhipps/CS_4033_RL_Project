import gymnasium as gym
import random 
import matplotlib.pyplot as plt
import numpy as np

class SARSA_LAMBDA():
    def __init__(self, alpha, epsilon, Lambda):
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = 0.9
        self.Lambda = Lambda
        
        self.wins = []
        
        # Initialize Q(s, a) arbitrarily for all (s, a) within S x A
        # arbitrarily set them between -1 and 1
        self.q_values = {}       
        for hand in range(4, 32):
           for dealer in range(1, 11):
               for ace in [0, 1]:
                   for action in [0, 1]:
                       self.q_values[(hand, dealer, ace, action)] = random.uniform(0, 1)
                       
        self.e = {}       
        
        pass
    
    def update_wins(self, reward):
        self.wins.append(reward)

 ##############################
 #### reset s(s, a) to 0 ######
 ##############################
 
    def reset_e(self):
        for hand in range(4, 32):
            for dealer in range(1, 11):
                for ace in [0, 1]:
                    for action in [0, 1]:
                        self.e[(hand, dealer, ace, action)] = 0
                        
 ##############################
    #### get e(s, a) ######
 ##############################         
    def get_e(self, state, action):
        hand = state[0]
        dealer = state[1]
        ace = state[2]
        
        return self.e[(hand, dealer, ace, action)]
                    

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

    ##########################
    ####### Updating #########
    ##########################
    
    def updating_delta(self, reward, state_next, action_next, state, action):
        hand = state[0]
        dealer = state[1]
        ace = state[2]
        
        hand_next = state_next[0]
        dealer_next = state_next[1]
        ace_next = state_next[2]
        
        q_sa = self.q_values[(hand, dealer, ace, action)]
        q_sa_next = self.q_values[(hand_next, dealer_next, ace_next, action_next)]
    
        # delta = r + gamma * Q(s', a') - Q(s, a)
        delta = reward * self.gamma * q_sa_next - q_sa
        
        return delta
    
    def updating_Q(self, state, action, delta):
        hand = state[0]
        dealer = state[1]
        ace = state[2]
        
        q_sa = self.q_values[(hand, dealer, ace, action)]
        e_sa = self.e[(hand, dealer, ace, action)]
        
        # Q(s, a) = Q(s, a) + alpha * delta * e(s, a)
        q_sa = q_sa + self.alpha * delta * e_sa
        
        return q_sa
    
    # second update type
    def updating_e(self, state, action):
        hand = state[0]
        dealer = state[1]
        ace = state[2]
        
        e_sa = self.e[(hand, dealer, ace, action)]
        
        #e(s, a) = gamma * lambda * e(s, a)
        e_sa = self.gamma * self.Lambda * e_sa
        
        return e_sa
    
##############################################################
###                        Plotting                        ###
##############################################################

def plot_learning_curve(wins):
    avg_wins = [sum(wins[:i+1]) / len(wins[:i+1]) for i in range(len(wins))]
    plt.plot(range(1, len(wins) + 1), avg_wins)
    plt.xlabel('Episodes')
    plt.ylabel('Average Win Rate')
    plt.title('Average Win Rate over Episodes')
    plt.show()
    
    
def calculate_mean_std(wins):
    mean_wins = np.mean(wins)
    std_wins = np.std(wins)
    print("Average wins: ", mean_wins)
    print("Standard Deviation of wins: ", std_wins)

##############################################################
##############################################################

def generate_episode(agent):
    
    done = False
    
    # Initialize e(s, a) = 0 for all s(s, a)
    agent.reset_e()
    
    # Initalize s 
    state = env.reset()[0]
    
    # Initalize a
    action = agent.policy(state)
    
    while not done:
        
        # Take action a, observe r, and next state s'
        state_next, reward, terminated, truncated, info = env.step(action)
        
        # Choose A' from S' using policy derived from Q
        action_next = agent.policy(state_next)
        
        # delta = r + gammaQ(s', a') - Q(s, a)
        delta = agent.updating_delta(reward, state_next, action_next, state, action)
        
        # e(s, a) = e(s, a) + 1 
        e_sa = agent.get_e(state, action)
        hand = state[0]
        dealer = state[1]
        ace = state[2]
        
        agent.e[(hand, dealer, ace, action)] = e_sa + 1
        
        # forall (s, a) within this episode do
        
        # Q(s, a) = Q(s, a) + alpha * delta * e(s, a)
        agent.q_values[(hand, dealer, ace, action)] = agent.updating_Q(state, action, delta)
        
            #e(s, a) = gamma * lambda * e(s, a)
        agent.e[(hand, dealer, ace, action)] = agent.updating_e(state, action)
        
    
        agent.update_wins(reward)
        
        # s = s'
        state = state_next
        # a = a'
        action = action_next
        
        # until S is terminal        
        done = terminated or truncated 
    
##############################################################
##############################################################


if __name__ == '__main__':
    env = gym.make("Blackjack-v1", sab=True)
    
    MAX_EPISODES = 10000
    alpha = 0.05
    epsilon = 0.1
    Lambda = 0.55
    
    agent = SARSA_LAMBDA(alpha, epsilon, Lambda)
    
    for _ in range(MAX_EPISODES):
        
        generate_episode(agent)
    
    
   # generate_episode(agent)
    
    print("Finished")
    calculate_mean_std(agent.wins)
    plot_learning_curve(agent.wins)
    
    
    
    
    
