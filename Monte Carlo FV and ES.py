import gymnasium as gym
import random 
import matplotlib.pyplot as plt
import numpy as np

CONST_WINS_FOLDER = "results/wins/"


"""
In First-vist MC we're just estimating V given v policy
So we just need V(s) and Returns(S)

For now, just have a given policy as it can be adjusted in a better algorithm.

"""
class FV_MC():
    def __init__(self):
        
        # Values for each state
        self.v_values = {}
        
        # Returns for each state
        self.returns = {}
        
        self.wins = []
        
        pass

#### Getters ####
    
    def get_values(self):
        return self.v_values
    
    def get_returns(self):
        return self.returns
 
    
####policy####

    def policy(self, state: tuple[int, int, bool]) -> int:
       agent_hand = state[0]

       if (agent_hand < 20):
           return 1
       else:
           return 0
     
    
 ######update######
 
 
    def update(self, episode: list):
       G = 0

       learned_reward = 0

       for t in range(len(episode) - 1, 0, -3):
           state = episode[t - 2]
           action = episode[t - 1]
           reward = episode[t]

           G = G + reward

           #check if new state == old state, if so continue
           if state in episode[: t-2]:
               continue
           
           #else if already here
           if state in self.returns:
               self.returns[state].append(G)
           
           # but not in returns then put G there
           else:
               self.returns[state] = [G]


class Monte_Carlo_ES():
    def __init__(self, epsilon):
        
        self.epsilon = epsilon
        
        self.wins = []
        
        # initialize policy values for each state
        self.polices = {}
        for hand in range(4, 32):
            for dealer in range(1, 11):
                for ace in [0, 1]:
                    self.polices[(hand, dealer, ace)] = random.choice([0, 1])
        
        # initialize q values and returns for each state and action
        self.q_values = {}
        self.returns = {}
        for hand in range(4, 32):
           for dealer in range(1, 11):
               for ace in [0, 1]:
                   for action in [0, 1]:
                       self.q_values[(hand, dealer, ace, action)] = 0
                       self.returns[(hand, dealer, ace, action)] = []
        
        pass
    
#### Helping Functions ####

    # average returns
    def average_returns(self, hand, dealer, ace, action):
        average_return = None
            
        returns = self.returns[(hand, dealer, ace, action)]
            
        average_return = sum(returns) / len(returns)
                
        return average_return
        
    # arg max of q values gi
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

#### Epsilon Greedy Policy ####

    def policy(self, state):
        hand = state[0]
        dealer = state[1]
        ace = state[2]
        
        if random.random() < self.epsilon:
            #return random action
            return random.randint(0, 1)
        
        else:
            return self.argmax_a_q(hand, dealer, ace)
            

#### Update ####

    def update(self, episode):
        
        G = 0
        
        for t in range(len(episode) - 1, 0, -3):
            state = episode[t - 2]
            hand = state[0]
            dealer = state[1]
            ace = state[2]
            
            action = episode[t - 1]
            reward = episode[t]
            
            G = G + reward
            
            if state in episode[: t-2]:
                continue
            
            # Append G to Returns(St, At)
            self.returns[(hand, dealer, ace, action)].append(G)
            
            # Q(St, At) average(Returns(St, At))
            self.q_values[(hand, dealer, ace, action)] = self.average_returns(hand, dealer, ace, action)

            # pi(St) = argmaxa Q(St, a)
            self.polices[(hand, dealer, ace)] = self.argmax_a_q(hand, dealer, ace)
            
                    

########################################################
########################################################

def gen_episode(agent) -> list:
    state = env.reset()[0]
    done = False
    
    episode = []
    
    # play one episode
    while not done:
        episode.append(state)
        action = agent.policy(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        
        episode.append(action)
        episode.append(reward)
        
        done = terminated or truncated
        state = next_state
        
    return episode


########################################################
#################### Plotting ##########################
########################################################


def plot_learning_curve(wins):
    avg_wins = [sum(wins[:i+1]) / len(wins[:i+1]) for i in range(len(wins))]
    plt.plot(range(1, len(wins) + 1), avg_wins)
    plt.xlabel('Episodes')
    plt.ylabel('Average Win Rate')
    plt.title('Average Win Rate over Episodes')
    plt.show()


def plot_learning_curve_no_mean(wins):
   # avg_wins = [sum(wins[:i+1]) / len(wins[:i+1]) for i in range(len(wins))]
    plt.plot(range(1, len(wins) + 1), wins)
    plt.xlabel('Episodes')
    plt.ylabel('Average Win Rate')
    plt.title('Average Win Rate over Episodes')
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
######################## Main ##########################
########################################################
if __name__ == '__main__':
    env = gym.make("Blackjack-v1", sab=True)
    
    algorithm = input("Enter algorithm: ")
    MAX_EPISODES = 10000
    num_games = 50

    if (algorithm == 'FV' or algorithm == 'fv'):
        
        for i in range(num_games):
            discount_factor = 1.0
        
            agent = FV_MC()
            
            wins = []
            
            for _ in range(MAX_EPISODES):
                
                episode = gen_episode(agent)
        
                # Apply First Visit to V(S) and Returns(S) based on episode
                agent.update(episode)
                
                reward = episode[2]
                agent.wins.append(reward)
        
            values = {state : sum(agent.get_returns()[state]) / len(agent.get_returns()[state]) for state in agent.get_returns()}
            plot_learning_curve(agent.wins)
            calculate_mean_std(agent.wins)
            #plot_learning_curve_no_mean(agent.wins)
            
            #save to csv in wins
            save_to_csv_wins(agent.wins, "MC_FV_ep_" + str(MAX_EPISODES) + "_" + str(i))
        

    if (algorithm == 'ES' or algorithm == 'es'):
        for i in range(num_games):
            epsilon = 0.1
            
            agent = Monte_Carlo_ES(epsilon)
            
            for _ in range(MAX_EPISODES):
                
                episode = gen_episode(agent)
                agent.update(episode)
                
                reward = episode[2]
                
                agent.wins.append(reward)
            
            plot_learning_curve(agent.wins)
            calculate_mean_std(agent.wins)
            #plot_learning_curve_no_mean(agent.wins)
            
            #save to csv in wins
            save_to_csv_wins(agent.wins, "MC_ES_ep_" + str(MAX_EPISODES) + "_" + str(i))
        
        
    
        
        
        
        
        


   
