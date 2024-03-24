import matplotlib as plt
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
from numpy import genfromtxt
import statistics as st


CONST_WINS_FOLDER = 'results/wins/'

num_games = 500

num_episodes = 1000
number_eps = x = np.arange(1, num_episodes + 1)

csv = ".csv"
colors = {'blue', 'green', 'orange', 'red'}

############################################################
######################## Data ##############################

### Wins ###



    # Monte Carlo First Visit
MC_FV_wins = genfromtxt(CONST_WINS_FOLDER + "MC_FV_ep_" + str(num_episodes) + csv, delimiter=',')
    
    # Monte Carlo ES
MC_ES_wins = genfromtxt(CONST_WINS_FOLDER + "MC_ES_ep_" + str(num_episodes) + csv, delimiter=',')
    
    # SARSA
SARSA_wins = genfromtxt(CONST_WINS_FOLDER + "SARSA_ep_" + str(num_episodes) + csv, delimiter=',')
    
    # SARSA(lambda)
SARSA_LAMBDA_wins = genfromtxt(CONST_WINS_FOLDER + "SARSA_LAMBDA_ep_" + str(num_episodes) + csv, delimiter=',')
    
SARSA_DECAY_wins = genfromtxt(CONST_WINS_FOLDER + "SARSA_DECAY_ep_" + str(num_episodes) + csv, delimiter=',')

################################
######## Plot Rewards ##########
################################

def plot_rewards():
    
    line1 = MC_FV_wins
    line1_label = "MC"
    
    line2 = MC_ES_wins
    line2_label = "ES"
    
    line3 = SARSA_wins
    line3_label = "SARSA"
    
    line4 = SARSA_LAMBDA_wins
    line4_label = "SARSA(lambda)"
    
    
    avg_wins_line1 = [sum(line1[:i+1]) / len(line1[:i+1]) for i in range(len(line1))]
    avg_wins_line2 = [sum(line2[:i+1]) / len(line2[:i+1]) for i in range(len(line2))]
    avg_wins_line3 = [sum(line3[:i+1]) / len(line3[:i+1]) for i in range(len(line3))]
    avg_wins_line4 = [sum(line4[:i+1]) / len(line4[:i+1]) for i in range(len(line4))]
    
   
    # Plot rewards over number of episodes
    plt.plot(number_eps, avg_wins_line1, label = line1_label) 
    plt.plot(number_eps, avg_wins_line2, label = line2_label) 
    plt.plot(number_eps, avg_wins_line3, label = line3_label)
    plt.plot(number_eps, avg_wins_line4, label = line4_label) 
    plt.plot(color=colors)
    
    plt.title("Rewards over " + str(num_episodes) + " episodes")
    plt.xlabel("Number of Episodes")
    plt.ylabel("Average Reward Accumulated")
    plt.legend() 
    plt.show()


def plot_percentage_wins():
    
    # Use barplot to see how many wins were accumulated 
  
    print("Winning percent in " + str(num_episodes) + " episodes")
    
    bar1 = len([a for a in MC_FV_wins if a > 0])
    bar1_total = (bar1 / num_episodes ) * 10 ** 2
    
    print("%.2f" % bar1_total + " %")
    
    bar2 = len([a for a in MC_ES_wins if a > 0])
    bar2_total = bar2 / num_episodes * 10 ** 2
    
    print("%.2f" % bar2_total + " %")
    
    bar3 = len([a for a in SARSA_wins if a > 0])
    bar3_total = bar3 / num_episodes * 10 ** 2
    
    print("%.2f" % bar3_total + " %")
    
    bar4 = len([a for a in SARSA_LAMBDA_wins if a > 0])
    bar4_total = bar4 / num_episodes * 10 ** 2
    
    print("%.2f" % bar4_total + " %")
    
    # Barplot 
    # Your data
    data = {
        "Algorithm": ["MC First Visit", "MC Explore Start", "SARSA", "SARSA(Lambda)"],
        "percentages": [bar1_total, bar2_total, bar3_total, bar4_total]
    }
    
    df = pd.DataFrame(data)
    
    # Create bar plot
    plt.figure(figsize=(8, 6))
    bars = plt.bar(df['Algorithm'], df['percentages'], color=colors)
    plt.xlabel('Algorithm')
    plt.ylabel('Percentages')
    plt.title('Percentages by Algorithm')
    plt.xticks(rotation=45)
    
    # Add numbers on top of bars
    for bar, color in zip(bars, colors):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), va='bottom', color=color)
    
    plt.tight_layout()
    
    # Show plot
    plt.show()
        

def percentage_wins(name_algo):
    
    list_of_percentages = []
    
    for i in range(num_games):
        wins = genfromtxt(CONST_WINS_FOLDER + name_algo + str(num_episodes) + "_" + str(i) + csv, delimiter=',')
        bar1 = len([a for a in wins if a > 0])
        bar1_total = (bar1 / num_episodes ) * 10 ** 2
        
        list_of_percentages.append(bar1_total)
        
    
    max_win_percent = max(list_of_percentages)
    min_win_percent = min(list_of_percentages)
    avg_win_percent = st.mean(list_of_percentages)
    
    print(name_algo + str(num_episodes))
    print(max_win_percent)
    print(min_win_percent)
    print(avg_win_percent)
    

#plot_percentage_wins()

#plot_rewards()

sarsa_decay = "MC_ES_ep_"

list_wins = percentage_wins(sarsa_decay)





