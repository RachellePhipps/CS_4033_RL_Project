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
    state = env.reset()[0]
    done = False

    episode = []
    episode.append(state)

    while not done:
        action = 0
        state, reward, done, truncated, info = env.step(action)
        episode.append(action)
        episode.append(reward)
        episode.append(state)

    return episode  


if __name__ == '__main__':
    env = gym.make("Blackjack-v1", sab=True, render_mode="human")
    
    for _ in range(100):
        episode = play_round()
        print(episode)
