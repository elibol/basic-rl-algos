# Author: Melih Elibol
# Description: Models which may be used for Markov decision processes.


import numpy as np

# This is the MDP approximation
# We can use the same MDP for Thompson;
# the only real purpose it serves is accurate
# policy computation given some approximate MDP,
# which only depends on the probability value being set properly,
# and the reward.


class ApproxMDP(object):

    def __init__(self, **kwargs):
        super(ApproxMDP, self).__init__()

        self.num_states = kwargs["num_states"]
        self.num_actions = kwargs["num_actions"]

        self.states = {}

    def update_state(self, state, action, next_state, reward):
        approx_state = self.get_state(state)
        approx_state.update_action_state(action, next_state, reward)

    def set_terminal_state(self, state):
        approx_state = self.get_state(state)
        approx_state.is_terminal = True

    def get_state(self, state, init_reward=0.0):
        if state not in self.states:
            self.states[state] = ApproxState(state_index=state,
                                             num_actions=self.num_actions,
                                             init_reward=init_reward)
        return self.states[state]


class ApproxState(object):

    def __init__(self, **kwargs):
        self.state_index = kwargs["state_index"]
        self.num_actions = kwargs["num_actions"]
        self.init_reward = kwargs["init_reward"]

        self.data = {}
        self.is_terminal = False

    def get_action(self, action):
        if action not in self.data:
            self.data[action] = {
                "next_states": {},
                "count": 0.0
            }
        return self.data[action]

    def get_action_state(self, action, state):
        action_struct = self.get_action(action)
        next_states = action_struct["next_states"]
        if state not in next_states:
            next_states[state] = {
                "count": 0.0,
                "probability": 0.0,
                "reward": 0.0
            }

        # import pdb; pdb.set_trace()
        return next_states[state]

    def update_action_state(self, action, next_state, reward):
        action_struct = self.get_action(action)
        action_struct["count"] += 1.0

        action_state = self.get_action_state(action, next_state)
        action_state["count"] += 1.0
        action_state["reward"] = reward

        return action_state["count"]


class RMAXMDP(ApproxMDP):

    def __init__(self, **kwargs):
        super(RMAXMDP, self).__init__(**kwargs)

        self.min_visits = kwargs["min_visits"]
        self.rmax = kwargs["rmax"]

        # init states
        for i in range(self.num_states):
            state = self.get_state(i, self.rmax)

    def state_known(self, state):
        return self.get_state(state).known

    def update_known(self, state):
        approx_state = self.get_state(state)
        if not approx_state.known:
            approx_state.update_known()
        return approx_state.known

    def get_state(self, state, init_reward=0.0):
        if state not in self.states:
            self.states[state] = RMAXState(state_index=state,
                                           num_actions=self.num_actions,
                                           init_reward=init_reward,
                                           min_visits=self.min_visits)
        return self.states[state]

    def update_probabilities(self):
        for approx_state in self.states.values():
            approx_state.compute_transitions()


class RMAXState(ApproxState):

    def __init__(self, **kwargs):
        super(RMAXState, self).__init__(**kwargs)

        self.min_visits = kwargs["min_visits"]

        # could use bit flags to figure out whether state is known
        # self.known_bits = 1 + 2 + 4 + 8
        self.known = False

        self.init_actions(self.init_reward)

    def init_actions(self, init_reward):
        for action_index in range(self.num_actions):
            state_struct = self.get_action_state(action_index, "special_state")
            state_struct["probability"] = 1.0
            state_struct["count"] = 1.0
            state_struct["reward"] = init_reward

    def update_known(self):
        # this can be optimized by maintaining bit flags
        # for actions that have reached the limit
        known = True
        for action_struct in self.data.values():
            if action_struct["count"] < self.min_visits:
                known = False
                break
        self.known = known
        return known

    def compute_transitions(self):
        for action_struct in self.data.values():
            total_count = action_struct["count"]
            if total_count == 0:
                continue
            # add 1.0 to compute proper distribution for rmax state
            for next_state in action_struct["next_states"].values():
                next_state["probability"] = next_state["count"] / (total_count + 1.0)


class BaysianMDP(ApproxMDP):

    def __init__(self, **kwargs):
        super(BaysianMDP, self).__init__(**kwargs)

        self.reward_mean = kwargs["reward_mean"]
        self.reward_std = kwargs["reward_std"]
        self.dirichlet_param = kwargs["dirichlet_param"]

        self.states = {}

        # init states
        for i in range(self.num_states):
            # initial reward is ignored at this level.
            state = self.get_state(i, 0.0)

    def get_state(self, state, init_reward=0.0):
        if state not in self.states:
            self.states[state] = BaysianState(state_index=state,
                                              init_reward=init_reward,
                                              num_actions=self.num_actions,
                                              num_states=self.num_states,
                                              dirichlet_param=self.dirichlet_param,
                                              reward_mean=self.reward_mean,
                                              reward_std=self.reward_std)
        return self.states[state]

    def update_probabilities(self):
        for approx_state in self.states.values():
            approx_state.compute_transitions()


class BaysianState(ApproxState):

    def __init__(self, **kwargs):
        super(BaysianState, self).__init__(**kwargs)

        self.reward_mean = kwargs["reward_mean"]
        self.reward_std = kwargs["reward_std"]
        self.dirichlet_param = kwargs["dirichlet_param"]
        self.num_states = kwargs["num_states"]

        self.init_actions(self.init_reward)

    def init_actions(self, init_reward):
        for action_index in range(self.num_actions):
            for state_index in range(self.num_states):
                state_struct = self.get_action_state(action_index, state_index)
                 # draw from dirichlet
                state_struct["count"] = self.dirichlet_param
                state_struct["reward"] = self.sample_reward()

    def compute_transitions(self):

        for action_index in range(self.num_actions):
            counts = np.ndarray(shape=( self.num_states))
            for state_index in range(self.num_states):
                state_struct = self.get_action_state(action_index, state_index)
                counts[state_index] = state_struct["count"]

            distribution = self.sample_dirichlet(counts)
            for state_index in range(self.num_states):
                state_struct = self.get_action_state(action_index, state_index)
                state_struct["probability"] = distribution[state_index]

    def sample_reward(self):
        sample = np.random.normal(self.reward_mean, self.reward_std)
        return sample

    def sample_dirichlet(self, nparray):
        return np.random.dirichlet(nparray)

if __name__ == "__main__":
    _mdp = RMAXMDP(min_visits=3)
    _mdp.update_probabilities()
