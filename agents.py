# Author: Melih Elibol
# Description: Reinforcement learning algorithms with consistent interface.

import numpy as np
from policy import EpsilonGreedyPolicy

from mdps import *


class AgentFactory(object):

    def __init__(self, agent_cls, policy_cls):
        self.agent_cls = agent_cls
        self.policy_cls = policy_cls

    def get_agent(self, num_states, num_actions, epsilon, gamma, alpha, agent_args):
        agent = self.agent_cls(num_states=num_states, num_actions=num_actions,
                               epsilon=epsilon, alpha=alpha, gamma=gamma,
                               policy_cls=self.policy_cls, agent_args=agent_args)
        return agent


class AgentBase(object):

    def __init__(self, num_states, num_actions, epsilon, gamma, alpha, policy_cls, agent_args):
        self.num_states = num_states
        self.num_actions = num_actions
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha

        self.policy_cls = policy_cls
        self.agent_args = agent_args
        self.state = None
        self.action = None

    def interact(self, reward, state, state_is_terminal):
        raise Exception("interact not implemented.")


class SarsaAgent(AgentBase):

    def __init__(self, **kwargs):
        super(SarsaAgent, self).__init__(**kwargs)
        self.policy = self.policy_cls(self.num_actions, self.epsilon)
        self.V = {}
        self.alpha_mode = self.agent_args["alpha_mode"]
        self.init_reward = self.agent_args["init_reward"]
        self.iterations = 0.0

    def interact(self, reward, next_state, next_state_is_terminal):
        next_action = self.get_action(next_state)

        if reward is None:
            self.state = next_state
            self.action = next_action
            return self.action

        self.iterations += 1.0

        if self.alpha_mode is "decay":
            self.alpha = 1.0/self.iterations
        elif self.alpha_mode is "min_decay":
            self.alpha = min(0.5, 10.0/self.iterations)
        elif self.alpha_mode is "constant":
            pass
        else:
            pass

        # update
        Q_s = self.get_V(self.state)[self.action]
        Q_spp = self.get_Q_spp(next_state, next_action)
        delta = reward + self.gamma*Q_spp - Q_s

        Q_s_next = Q_s + self.alpha*delta
        self.V[self.state][self.action] = Q_s_next
        # self.V[self.state][self.action] += self.alpha*delta

        # we're done with this episode; reset iterations
        if next_state_is_terminal:
            self.iterations = 0.0

        self.state = next_state
        self.action = next_action
        return self.action

    def get_action(self, state):
        # derive / compute policy based on Q
        action_values = self.get_V(state)
        self.policy.update(state, action_values)
        return self.policy.get_action(state)

    # lazy-initialize Q(s,a) for all a in A(s) to self.init_reward
    def get_V(self, s):
        if s not in self.V:
            self.V[s] = [self.init_reward]*self.num_actions
        return self.V[s]

    def get_Q_spp(self, next_state, next_action):
        return self.get_V(next_state)[next_action]


class QLearningAgent(SarsaAgent):

    def __init__(self, **kwargs):
        super(QLearningAgent, self).__init__(**kwargs)

    def get_Q_spp(self, next_state, next_action):
        return max(self.get_V(next_state))

# This just requires an MDP that subclasses
# ApproxMDP, and uses a state class that's
# a subclass of ApproxState
class ValueIterationAgent(object):

    def __init__(self, num_actions, gamma, theta, policy_cls):
        self.num_actions = num_actions
        self.gamma = gamma
        self.theta = theta
        self.policy_cls = policy_cls

        # retain v because the last set of state values
        # will be closer to the optimal value function
        # than arbitrary values.
        self.v = {}

    def new_policy(self):
        return self.policy_cls(self.num_actions)

    def compute_policy_by_mdp(self, mdp):

        v = self.v

        def getv(s, init=0.0):
            if s not in v:
                v[s] = init
            return v[s]

        # Using this structure to compute new policy
        # careful with this, though. special_state should still have reward 0.
        # this may not accurately represent value given by v[s], as v[s] is maximum of all actions.
        # but theoretically, it's ok, because v(s') is value of next state given optimal choice.
        q = {}

        def setq(s, a, value):
            if s not in q:
                q[s] = {}
            q[s][a] = value

        while True:
            delta = 0.0

            for state_index, approx_state in mdp.states.iteritems():
                max_value = float("-inf")

                for action_index, action in approx_state.data.iteritems():
                    next_states = action["next_states"]
                    state_value = 0.0
                    for next_state_id, next_state in next_states.iteritems():
                        # p(s'|s,a)
                        prob = next_state["probability"]
                        # r(s,a,s')
                        reward = next_state["reward"]
                        # gamma*v(s')
                        next_value = self.gamma*getv(next_state_id)
                        state_value += prob*(reward + next_value)

                    max_value = max(max_value, state_value)

                    # set terminal state action values to 0
                    if approx_state.is_terminal:
                        setq(state_index, action_index, 0)
                    else:
                        setq(state_index, action_index, state_value)

                # update delta
                delta = max(delta, abs(max_value - getv(state_index)))
                # update value
                v[state_index] = max_value

            if delta < self.theta:
                break

        policy = self.new_policy()
        for s_index, s_data in q.iteritems():
            policy.update(s_index, s_data)

        return policy


class RMAXAgent(AgentBase):

    def __init__(self, **kwargs):
        super(RMAXAgent, self).__init__(**kwargs)
        self.steps = 0
        self.max_steps = self.agent_args["max_steps"]
        self.min_visits = self.agent_args["min_visits"]
        self.theta = self.agent_args["theta"]
        self.rmax = self.agent_args["rmax"]

        self.mdp = RMAXMDP(min_visits=self.min_visits,
                           num_actions=self.num_actions,
                           num_states=self.num_states,
                           rmax=self.rmax)

        self.vi_agent = ValueIterationAgent(self.num_actions, self.gamma, self.theta, self.policy_cls)

        self.policy = None
        self.compute_policy()

    def interact(self, reward, next_state, next_state_is_terminal):
        # import pdb; pdb.set_trace()

        # pick action using current optimal policy
        next_action = self.policy.get_action(next_state)

        if reward is None:
            self.state = next_state
            self.action = next_action
            return self.action

        # invalidation flag
        invalidate = False

        # update state
        self.mdp.update_state(self.state, self.action, next_state, reward)
        preknown = self.mdp.state_known(self.state)
        known = self.mdp.update_known(self.state)

        if next_state_is_terminal:
            self.mdp.set_terminal_state(next_state)

        if (not preknown) and known:
            invalidate = True

        self.steps += 1
        if self.steps >= self.max_steps:
            self.steps = 0
            invalidate = True

        if invalidate:
            self.compute_policy()

        self.state = next_state
        self.action = next_action

        return self.action

    # just use value iteration
    def compute_policy(self):
        # update transition probabilities
        self.mdp.update_probabilities()
        self.policy = self.vi_agent.compute_policy_by_mdp(self.mdp)


class ThompsonAgent(AgentBase):

    def __init__(self, **kwargs):
        super(ThompsonAgent, self).__init__(**kwargs)
        self.max_steps = self.agent_args["max_steps"]
        self.theta = self.agent_args["theta"]
        self.reward_mean = self.agent_args["reward"]["mean"]
        self.reward_std = self.agent_args["reward"]["std"]
        self.dirichlet_param = self.agent_args["dirichlet"]

        self.steps = 0

        self.mdp = BaysianMDP(num_actions=self.num_actions,
                              num_states=self.num_states,
                              reward_mean=self.reward_mean,
                              reward_std=self.reward_std,
                              dirichlet_param=self.dirichlet_param)

        self.vi_agent = ValueIterationAgent(self.num_actions,
                                            self.gamma,
                                            self.theta,
                                            self.policy_cls)

        self.policy = None
        self.compute_policy()

    def interact(self, reward, next_state, next_state_is_terminal):

        # pick action using current optimal policy
        next_action = self.policy.get_action(next_state)

        if reward is None:
            self.state = next_state
            self.action = next_action
            return self.action

        # invalidation flag
        invalidate = False

        # update state
        self.mdp.update_state(self.state, self.action, next_state, reward)

        if next_state_is_terminal:
            self.mdp.set_terminal_state(next_state)

        self.steps += 1
        if self.steps >= self.max_steps:
            self.steps = 0
            invalidate = True

        if invalidate:
            self.compute_policy()

        self.state = next_state
        self.action = next_action

        return self.action

    # just use value iteration
    def compute_policy(self):
        # update transition probabilities
        self.mdp.update_probabilities()
        self.policy = self.vi_agent.compute_policy_by_mdp(self.mdp)
