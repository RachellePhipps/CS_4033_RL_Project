import gymnasium as gym
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class BlackjackAgent():
    def __init__(self):
        pass

    def policy(self, state):
        score, dealer_score, usable_ace = state[0]
        #return random.choice([0,1])
        if (score >= 17):
            return 0
        
        else:
            return 1


def generate_episode(agent):
    state_n = env.reset()
    done = False
    episode = []

    while not done:
        action_n = agent.policy(state_n)
        state_n_1, reward_n, done, truncated, info = env.step(action_n)

        episode.append(state_n)
        episode.append(action_n)
        episode.append(reward_n)
        episode.append(state_n_1)

    return episode

def plot_values(values):
   # get x, y, and z from values
    x = []
    y = []
    z = []

    for state in values:
        score, dealer_score, usable_ace = state
        # just care about agent scoe and dealer's for now
        x.append(score)
        y.append(dealer_score)
        z.append(values[state])

   # print(x)
   # print(y)
   # print(z)
    
    # to Add a color bar which maps values to colors.
    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d')
    surf=ax.plot_trisurf(y, x, z, cmap=plt.cm.inferno, linewidth=0.2)
    fig.colorbar( surf, shrink=0.5, aspect=5)
    plt.show()


if __name__ == '__main__':
    env = gym.make("Blackjack-v1", sab=True)
    
    max_episodes = 1000
    # Initialize 
    """ V(s) within R for all s in S
    Returns <- empty list
    """ 
    values = {}
    returns = {}

    agent = BlackjackAgent()

    for _ in range(max_episodes):
        # generate episode following pi: S0 -> A0 -> R1 -> S1 ....
        episode = generate_episode(agent) # returns s_0, a_0, r_0, s_1, ...s_n etc
        
        G = 0

        # FIXME change to understand s_0, a_0, r_0, s_1, ...s_n etc
        for i in range(len(episode) - 1, 0, -3):
            previous_state = episode[0][0]
            action = episode[1]
            reward = episode[2]
            next_state = episode[3]

            G = G + reward

            #check if new state == old state, if so continue
            if next_state == previous_state:
                continue
            
            #else if already here
            if previous_state in returns:
                returns[previous_state].append(G)
            
            # but not in returns then put G there
            else:
                returns[previous_state] = [G]
 
    values = {state : sum(returns[state]) / len(returns[state]) for state in returns}
    print(values)
    #plot_values(values)





            
                


