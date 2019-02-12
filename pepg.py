# -*- coding: utf-8 -*-
""" Agent class for Reinforcement Learning PA - Spring 2018

Details:
    File name:          my_agent.py
    Author(s):          TODO: fill in your own name(s) and student ID(s)
    Date created:       28 March 2018
    Date last modified: TODO: fill in
    Python Version:     3.4

Description:
    TODO: briefly explain which algorithm you have implemented and what this
    agent actually does

Related files:
    base_agent.py
"""
from base_agent import BaseAgent
import gym
import numpy as np
import itertools
from joblib import Parallel, delayed
import logging
from time import time
import matplotlib.pyplot as plt
from gym import logger


NUMBER_OF_THREADS = 3
logging.disable(logging.CRITICAL) # disable messages about env creation
logger.set_level(40)


class nnpolicy(object):
    def __init__(self, state_size, action_size, n_hidden=32):
        """
        Creates a policy that uses forward-pass neural network with two hidden layers
        :param n_features: Number of features of the state (and size of the input layer)
        :param n_actions: Number of actions (and size of the output layer)
        :param n_hidden1: Size of first hidden layer
        """
        self.action_size = action_size
        self.state_size = state_size
        self.n_hidden = n_hidden
    
    @staticmethod
    def unpack(self, theta): 
        """
        Unpacks 1D array to separate arrays with weights and biases for neural network in order: w, b, w2, b2, w3, b3
        :param theta: 1D array with all parameters for the policy
        :return: Unpacked parameters
        """
        shapes = [(self.state_size, self.n_hidden),
                  (1, self.n_hidden),
                  (self.n_hidden, self.action_size),
                  (1, self.action_size),]
        result = []
        start = 0
        for i, offset in enumerate(np.prod(shape) for shape in shapes):
            result.append(theta[start:start + offset].reshape(shapes[i]))
            start += offset
        return result
    
    def forward_pass(self, theta, state): 
        """
        Selects action for policy with given theta for a given state \pi_\theta(s)
        :param theta: Policy parametrization
        :param state: State
        :return: selected action
        """
        w, b, w2, b2 = self.unpack(self, theta)

        z = state.dot(w) + b
        a1 = np.tanh(z)

        z2 = a1.dot(w2) + b2
        a2 = np.tanh(z2)

        return np.argmax(a2)
    
    def get_number_of_parameters(self): 
        """
        Computes total number of parameters for policy
        :return:
        """
        return (self.state_size + 1) * self.n_hidden + (self.n_hidden + 1) * self.action_size

class MyAgent(BaseAgent):
    """ Parameter-exploring Policy Gradient
    """

    def __init__(self, *args, alpha_u=0.0001, alpha_sigma=0.00001, initial_u = 0.0, initial_sigma=0.5, history_size=50, population_size=600, test_iterations=20, **kwargs):
        """
        Creates an agent using Parameter-exploring Policy Gradient algorithm
        :param env: OpenAI gym environment to solve. Assuming continuous state space and discrete action space
        :param initial_u: Initial mu
        :param alpha_u: Learning rate for mu
        :param initial_sigma: Initial sigma
        :param alpha_sigma: Learning rate for sigma
        :param history_size: Size of history for the algorithm to consider
        :param population_size: Size of the population
        :param test_iterations: Number of iterations in test phase
        """        
        super().__init__(*args, **kwargs)
        self.state_size = 8
        self.action_size = 4
        self.test_iterations = test_iterations
        self.history_size = history_size
        self.validation_env = gym.make(self._wrapper._env.spec.id)
        
        self.policy = nnpolicy(self.state_size, self.action_size, n_hidden = 32)
        self.P = self.policy.get_number_of_parameters() #number of weights
        self.N = population_size
        
        self._total_reward_history = []
        self.val_history = []
        self.mean_history = []
        self.rolling_history = []

        self.u = np.repeat(initial_u, self.P) #mean of distribution
        self.sigma = np.repeat(initial_sigma, self.P) #standard deviation of distribution
        self.alpha_u = alpha_u #learning rate of mean
        self.alpha_sigma = alpha_sigma #learning rate of sigma
        
        self.b = 0.0 

    def initialise_episode(self):
        self._total_reward = 0
        return self._wrapper.reset()

    def select_action(self, *args):
        self.get_action_value
        pass
    
    def update_plot(self, run):
        """
        Makes the plot
        :param val_history: History of validation scores
        :param mean_history: History of mean scores for whole population
        :param rolling_history: 100 rolling average of validation scores
        """
        plt.clf()
        plt.title('Performance of Parameter-exploring Policy Gradient for run %s'%run)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.plot(self.val_history)
        plt.plot(self.mean_history)
        plt.plot(self.rolling_history)
        plt.legend(['validation', 'mean', '100 average'])
        plt.savefig('pepg-try%s.png'%run)
        plt.pause(0.05)

    def train(self):
        evaluation_start_time = time()
        #Create 2D array of N rows filled with probabilities drawn from normal distribution around u and sigma
        #Each row consists of P (number of weights) random samples
        theta = np.zeros((self.N, self.P))
        for n in range(self.N):
            theta[n,:] = np.random.normal(self.u,self.sigma)
        
        #Evaluate the rewards for each set of weights drawn from normal distribution around u and sigma
        #r = array of total episode reward for each row N in theta
        results = np.array(Parallel(n_jobs=NUMBER_OF_THREADS)(delayed(evaluate_policy)(self.policy, theta[n, :]) for n in range(self.N)))
        
        #Evaluate current policy with the mean u as weights
        val_score = evaluate_policy(self.policy, self.u, self.validation_env, render=False)
        
        #Compute history
        self.mean_history.append(np.mean(results)) #mean of rewards of N different initialisations
        self._total_reward_history.append(val_score) #reward of current policy
        self.val_history.append(val_score) #reward of current policy
        self.rolling_history.append(np.mean(self.val_history[-100:])) #mean of last 100 rewards
        
        #Add current run to plot
        #self.update_plot(run)
        
        #Calculations before updating policy
        T = theta.T - np.repeat(np.array([self.u]).T, self.N, axis=1) #distance of each probability sample to mean
        S = np.divide(np.square(T.T) - np.square(self.sigma), self.sigma).T
        
        results -= self.b
        results = results.T
        
        #update policy
        self.u += self.alpha_u * np.matmul(T,results)
        self.sigma += self.alpha_sigma * np.matmul(S,results)
        
        #update b: mean of last 50(history_size) rewards
        self.b = np.mean(self._total_reward_history[-self.history_size:])

        iteration_duration = time() - evaluation_start_time
        print('Iteration completed after {:2f} seconds with validation reward {}'.format(iteration_duration, val_score))
        return val_score

def evaluate_policy(policy, theta, env=None, render=False):
    """
    Evaluates policy with given parametrization theta.
    :param theta: Policy parametrization
    :param env: Env to act inside of. If none, a new one is created
    :param render: Renders world state?
    :return: Cumulative reward for single episode
    """
    if env is None:
        env = gym.make('LunarLander-v2')
    episode_reward = 0
    state = env.reset()
    while True:
        action = policy.forward_pass(theta, state)
        new_state, reward, done, _ = env.step(action)
        if render:
            env.render()
                
        state = new_state
        episode_reward += reward
            
        if done:
            return episode_reward
