import gymnasium as gym
import random

class Player():
    def __init__():
        pass

def play_round():
    state = env.reset()
    done = False
    
    while not done:
        action = random.choice([0,1])
        observation, reward, done, truncated, info = env.step(action)
        env.render()

    return reward



if __name__ == '__main__':
    env = gym.make("Blackjack-v1", sab=True, render_mode="human")
    
    print("Round 1 reward: ", play_round())
