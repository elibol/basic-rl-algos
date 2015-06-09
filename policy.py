# Author: Melih Elibol
# Description:

import numpy as np


class EpsilonGreedyPolicy(object):
    
    def __init__(self, num_actions,  epsilon=0):
        self.states = {}
        self.epsilon = epsilon
        self.action_len = num_actions
        self.action_range = range(self.action_len)
        
    # lazy-initialize states w/ uniform dist.
    def get_state(self, s):
        if s not in self.states:
            self.states[s] = [1.0/self.action_len]*self.action_len
        return self.states[s]
    
    # epsilon-greedy update.
    def update(self, s, action_values):
        action_dist = self.get_state(s)
        if sum(action_values) == 0:
            return
        
        # find indices that are maximal
        arg_max = []
        max_a = float("-inf")
        for i in self.action_range:
            val_a = action_values[i]
            if val_a > max_a:
                max_a = val_a
                arg_max = [i]
            elif val_a == max_a:
                arg_max.append(i)
        
        # distribute fraction of epsilon over low indices
        # give remaining prob. to hi indices
        num_hi = len(arg_max)
        num_low = self.action_len - len(arg_max)
        low_prob = self.epsilon/self.action_len
        hi_prob = (1.0 - low_prob*num_low)/num_hi
        for i in self.action_range:
            if i in arg_max:
                action_dist[i] = hi_prob
            else:
                action_dist[i] = low_prob
    
    def dist(self, s):
        return self.get_state(s)
    
    def get_action(self, s):
        action_dist = self.get_state(s)
        action = np.random.choice(self.action_len, 1, p=action_dist)
        return action[0]
