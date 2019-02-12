# -*- coding: utf-8 -*-
""" Lunar Lander environment wrapper for Reinforcement Learning PA - Spring 2018

Details:
    File name:          lunarlander_wrapper.py
    Author(s):          Paul Couzy (s1174347) & Eva van Weenen (s1376969)
    Date created:       28 March 2018
    Date last modified: TODO: fill in
    Python Version:     3.4

Description:
    Implementation of a wrapper for the Lunar Lander environment as presented in
    https://gym.openai.com/envs/LunarLander-v2/

Related files:
    wrapper.py
"""

from wrapper import Wrapper
import numpy as np
import pandas as pd


class LunarLanderWrapper(Wrapper):
    """ TODO: Add a description for your wrapper
    """

    _actions = [0,1,2,3]   # define list of actions (HINT: check LunarLander-v2 source code to figure out what those actions are)
    
    def __init__(self):
        super().__init__(env_name='LunarLander-v2', actions=self._actions)  # Don't change environment name
        print(self._actions)
        self._penalty = -100 #penalty for crashing
        """
        # Define the discretisation of the state vector
        # First define the value ranges
        x_lim = 1 #lunar_lander VIEWPORT_W
        y_lim = 1 #lunar_lander VIEWPORT_H
        velo_x_lim = 1 
        velo_y_lim = 1
        ang_lim = np.pi #in radians
        velo_ang_lim = 1

        # Then define the numbers of bins
        n_x_bins = 10
        n_y_bins = 10
        n_velo_x_bins = 10
        n_velo_y_bins = 10
        n_ang_bins = 10
        n_velo_ang_bins = 10
        
        #Create the bins
        ship_x_bins = pd.cut([-x_lim, x_lim], bins=n_x_bins, retbins=True)[1][1:-1]
        ship_y_bins = pd.cut([-y_lim, y_lim], bins=n_y_bins, retbins=True)[1][1:-1]
        ship_velo_x_bins = pd.cut([-velo_x_lim, velo_x_lim], bins=n_velo_x_bins, retbins=True)[1][1:-1]
        ship_velo_y_bins = pd.cut([-velo_y_lim, velo_y_lim], bins=n_velo_y_bins, retbins=True)[1][1:-1]
        ship_ang_bins = pd.cut([-ang_lim, ang_lim], bins=n_ang_bins, retbins=True)[1][1:-1]
        ship_velo_ang_bins = pd.cut([-velo_ang_lim, velo_ang_lim], bins=n_velo_ang_bins, retbins=True)[1][1:-1]
        
        leg1 = [False, True]
        leg2 = [False, True]
        
        # Set self._bins variable
        self._bins = [ship_x_bins, ship_y_bins, ship_velo_x_bins, ship_velo_y_bins, ship_ang_bins, ship_velo_ang_bins, leg1, leg2]
        
        # TODO: write the rest of your initialisation function
        """
   
    def get_bins(self):
        """ Returns a list of lists, such that for a state vector (x0, ..., xn),
        the zeroth element of the list contains the list of bins for variable
        x1, the first element of the list contains the list of bins for variable
        x2, and so on.
        """
        return self._bins
    
    def episode_over(self):
        """ Checks if the episode can be terminated because the maximum number
         of steps (200) has been taken """
        return True if self._number_of_steps >= 623 else False
    
    def solved(self, rewards):
        if (len(rewards) >= 100) and (sum(1 for r in rewards if r>=200) >= 10):
            return True
        #return True if reward >= 200 else False

    def penalty(self):
        return self._penalty
    
    # TODO: implement all other functions and methods needed for your wrapper
    
