from collections import defaultdict
import gymnasium as gym
import random
import matplotlib.pyplot as plt
import numpy as np

class Blackjack():
    def __init__(
    ):
        pass

def play_round():
    state = env.reset()

    print("state", state)
    done = False
    while not done:
        action = random.choice([0,1])
        print("action took: ", action)  
        observation, reward, done, truncated, info = env.step(action)
        print("step action: ", env.step(action))
        env.render()

    return reward


if __name__ == '__main__':
    env = gym.make("Blackjack-v1", sab=True, render_mode="human")
    
    print("Round 1 reward: ", play_round())
