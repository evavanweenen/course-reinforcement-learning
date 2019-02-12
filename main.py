# -*- coding: utf-8 -*-
""" Main script for Reinforcement Learning PA - Spring 2018

Details:
    File name:          main.py
    Author:             Anna Latour
    Date created:       19 March 2018
    Date last modified: 26 March 2018
    Python Version:     3.4

Description:
    This is the main file for running your experiments for the Practical
    Assignment for the LIACS Reinforcement Learning course of Spring 2018.

    It contains a couple of hints for how you can run experiments and store
    your results.

Related files:
    wrapper.py
    cartpole_wrapper.py
    lunarlander_wrapper.py
    base_agent.py
    qlearner.py
    my_agent.py
	policy_gradient.py
	pepg.py
How to run:
    $ python main.py -h
"""

import argparse
import pandas as pd
from timeit import default_timer as timer
#from qlearner import QLearner
#from cartpole_wrapper import CartPoleWrapperDiscrete



from lunarlander_wrapper import LunarLanderWrapper

"""
Choose on of the following agents, the pepg for the parameter estimation policy gradient 
or the monte_carlo for Monte-Carlo policy gradient


"""
from pepg import MyAgent
#from monte_carlo import MyAgent

# Parse the arguments
parser = argparse.ArgumentParser(description="Experiment parameters")
parser.add_argument('-e', '--episodes', type=int, default=5000,
                    help="The maximum number of episodes per run")
parser.add_argument('-r', '--runs', type=int, default=10,
                    help="The number of runs (repeats) of the experiment")
args = parser.parse_args()
num_episodes = args.episodes
num_runs = args.runs

################################################################################
#                                                                              #
#                              EXAMPLE: CART POLE                              #
#                                                                              #
################################################################################
"""
# Initialise result data structures
rewards_per_run = dict()
runtime_per_run = []

# For each run, train agent until environment is solved, or episode budget
# runs out:
for run in range(num_runs):
    # Initialise result helpers
    end_episode = num_episodes      # indicates in which run the environment was solved
    start = timer()
    rewards = [0.0] * num_episodes          # reward per episode

    # Initialise environment and agent
    wrapper = CartPoleWrapperDiscrete()
    agent = QLearner(wrapper=wrapper, seed=run)

    # For each episode, train the agent on the environment and record the
    # reward of each episode
    for episode in range(num_episodes):
        rewards[episode] = agent.train()
        # Check if environment is solved
        if wrapper.solved(rewards[:episode]):
            end_episode = episode
            break

    # Record and print performance
    runtime_per_run.append(timer() - start)
    rewards_per_run['run' + str(run)] = rewards
    if end_episode >= 99:
        print('average reward of last 100 episodes of run', run,
              '=', float(sum(rewards[-100:])) / 100)
    print('end episode # = ', end_episode)

    # Close environment
    wrapper.close()

# Store results
df_rewards = pd.DataFrame(rewards_per_run)
df_rewards.to_csv('cartpole_rewards.csv')

df_time = pd.DataFrame(runtime_per_run, columns=['time [s]'])
df_time.to_csv('cartpole_runtimes.csv')
"""

################################################################################
#                                                                              #
#                           ASSIGNMENT: LUNAR LANDER                           #
#                                                                              #
################################################################################



# Initialise result data structures
rewards_per_run = dict()
runtime_per_run = []

# For each run, train agent until environment is solved, or episode budget 
# runs out:
for run in range(num_runs):
    # Initialise result helpers
    end_episode = num_episodes  # indicates in which run the environment was solved
    start = timer()
    rewards = [0.0] * num_episodes  # reward per episode

    # Initialise environment and agent
    wrapper = LunarLanderWrapper()
 
    agent = MyAgent(wrapper=wrapper, seed=run)    
    
    # For each episode, train the agent on the environment and record the
    # reward of each episode
    for episode in range(num_episodes):
        #rewards[episode] = agent.train(episode)
        print("episode", episode)
        rewards[episode] = agent.train()
        # Check if environment is solved
        if wrapper.solved(rewards[:episode]):
            end_episode = episode
            break

    # Record and print performance
    runtime_per_run.append(timer() - start)
    rewards_per_run['run' + str(run)] = rewards
    if end_episode >= 99:
        print('average reward of last 100 episodes of run', run,
              '=', float(sum(rewards[-100:])) / 100)
    print('end episode # = ', end_episode)


    # Close environment
    wrapper.close()

# Store results
df_rewards = pd.DataFrame(rewards_per_run)
df_rewards.to_csv('lunarlander_rewards.csv')

df_time = pd.DataFrame(runtime_per_run, columns=['time [s]'])
df_time.to_csv('lunarlander_runtimes.csv')
