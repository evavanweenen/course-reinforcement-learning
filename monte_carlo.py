# -*- coding: utf-8 -*-
""" Agent class for Reinforcement Learning PA - Spring 2018

Details:
    File name:          my_agent.py
    Author(s):          Paul Couzy & Eva van Weenen
    Date created:       1 May 2018
    Date last modified: 14 May 2018
    Python Version:     3.4

Description:
  	Uses Policy gradient and a neural network with tensorflow to get actions for lunar lander
	Based on:
    https://leimao.github.io/article/REINFORCE-Policy-Gradient/
	problems of high variance
Related files:
    base_agent.py
"""

import random
from base_agent import BaseAgent
import numpy as np
import tensorflow as tf

class MyAgent(BaseAgent):
	""" Policy Gradient
	"""
	def __init__(self, *args, gamma=0.99, learning_rate=0.005, **kwargs):
		super().__init__(*args, **kwargs)
		tf.reset_default_graph()#clears all the stored variables for tensorflow after each run
		self._gamma = gamma # discount factor
		self.learning_rate = learning_rate
		self.state_size = 8 
		self.action_size = 4
		self.model= self.reinforce_policy_setup()#builds our model using tensorflow
		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())
        # Initialize episode replays used for caching game transitions in one single episode
		self.episode_observations = list() # observation feature list
		self.episode_actions = list() # one-hot encoded action
		self.episode_rewards = list() # immediate reward
		
	def reinforce_policy_setup(self):#setups the neural network or policy network
		with tf.name_scope('inputs'):
			self.tf_observations=tf.placeholder(tf.float32,[None, self.state_size], name = 'observations')
			self.tf_actions = tf.placeholder(tf.int32,[None,], name = 'num_actions')
			self.tf_values = tf.placeholder(tf.float32,[None,], name= 'state_values')

		fc1 = tf.layers.dense(
			inputs = self.tf_observations,
			units = 16,
			activation = tf.nn.tanh,  # tanh activation
			kernel_initializer = tf.random_normal_initializer(mean=0, stddev=0.3),
			bias_initializer = tf.constant_initializer(0.1),
			name='FC1', reuse=tf.AUTO_REUSE
		)

		# FC2
		fc2 = tf.layers.dense(
			inputs = fc1,
			units = 32,
			activation = tf.nn.tanh,  # tanh activation
			kernel_initializer = tf.random_normal_initializer(mean=0, stddev=0.3),
			bias_initializer = tf.constant_initializer(0.1),
			name='FC2', reuse=tf.AUTO_REUSE
		)

		# FC3
		logits = tf.layers.dense(
			inputs = fc2,
			units = self.action_size,
			activation = None,
			kernel_initializer = tf.random_normal_initializer(mean=0, stddev = 0.3),
			bias_initializer = tf.constant_initializer(0.1),
			name='FC3', reuse=tf.AUTO_REUSE
		)

		# Softmax
		self.action_probs = tf.nn.softmax(logits, name='action_probs')

		with tf.name_scope('loss'):
            # To maximize (log_p * V) is equal to minimize -(log_p * V)
            # Construct loss function mean(-(log_p * V)) to be minimized by tensorflow
			neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = self.tf_actions) # this equals to -log_p
			self.loss = tf.reduce_mean(neg_log_prob * self.tf_values)

		with tf.name_scope('train'):
			self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

	def Store_Transition(self, observation, action, reward):
        # Store game transitions used for updating the weights in the Policy Neural Network
		self.episode_observations.append(observation)
		self.episode_actions.append(action)
		self.episode_rewards.append(reward)

	def Clear_Episode_Replays(self):
        # Clear game transitions
		self.episode_observations = list()
		self.episode_actions = list()
		self.episode_rewards = list()

	def Calculate_Value(self):

        # The estimate of v(St) is updated in the direction of the complete return:
        # Gt = Rt+1 + gamma * Rt+2 + gamma^2 * Rt+3 + ... + gamma^(T-t+1)RT;
        # where T is the last time step of the episode.
		state_values = np.zeros_like(self.episode_rewards)
		state_values[-1] = self.episode_rewards[-1]
		for t in reversed(range(0, len(self.episode_rewards)-1)):
			state_values[t] = self._gamma * state_values[t+1] + self.episode_rewards[t]

        # Normalization to help the control of the gradient estimator variance
		state_values -= np.mean(state_values)
		state_values /= np.std(state_values)
		return state_values	

	def REINFORCE_FC_Train(self):

        # Train model using data from one episode
		inputs = np.array(self.episode_observations)
		state_values = self.Calculate_Value()

        # Start gradient ascent
		_, train_loss = self.sess.run([self.optimizer, self.loss], feed_dict = {
		self.tf_observations: np.vstack(self.episode_observations),
		self.tf_actions: np.array(self.episode_actions), 
		self.tf_values: state_values})

        # Clear episode replays after training for one episode
		self.Clear_Episode_Replays()

		return train_loss

	def select_action(self, observation):

        # Calculate action probabilities when given observation
		prob_weights = self.sess.run(self.action_probs, feed_dict = {self.tf_observations: observation[np.newaxis, :]})

        # Choose action according to the probabilities 
		action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())
		return action

	def train(self): 
		observation = self._wrapper.reset()#reset the environment
		episode_reward = 0 # reset reward
		while True:
			action = self.select_action(observation = observation)
			observation_next, reward, done, info = self._wrapper.step(action)
			self.Store_Transition(observation = observation, action = action, reward = reward)
			observation = observation_next
			episode_reward += reward

			if done:
				# Train on one episode
				train_loss = self.REINFORCE_FC_Train()
				return episode_reward	
