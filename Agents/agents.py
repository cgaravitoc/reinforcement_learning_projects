'''
Classes for implementing the learning methods.
'''
from random import randint, choice
import numpy as np
import random

class Agent :
    '''
    Defines the basic methods for the agent.
    '''

    def __init__(self, parameters:dict):
        self.parameters = parameters
        self.nS = self.parameters['nS']
        self.nA = self.parameters['nA']
        self.gamma = self.parameters['gamma']
        self.epsilon = self.parameters['epsilon']
        self.states = []
        self.actions = []
        self.rewards = [np.nan]
        self.dones = [False]
        self.policy = np.ones((self.nS, self.nA)) * 1/self.nA
        self.Q = np.zeros((self.nS, self.nA))

    def make_decision(self):
        '''
        Agent makes a decision according to its model.
        '''
        state = self.states[-1]
        weights = [self.policy[state, action] for action in range(self.nA)]
        action = random.choices(population = range(self.nA),\
                                weights = weights,\
                                k = 1)[0]
        return action

    def restart(self):
        '''
        Restarts the agent for a new trial.
        '''
        self.states = []
        self.actions = []
        self.rewards = [np.nan]
        self.dones = [False]

    def reset(self):
        '''
        Resets the agent for a new simulation.
        '''
        self.restart()
        self.policy = np.ones((self.nS, self.nA)) * 1/self.nA
        self.Q = np.zeros((self.nS, self.nA))

    def max_Q(self, s):
        '''
        Determines the max Q value in state s
        '''
        return max([self.Q[s, a] for a in range(self.nA)])

    def argmaxQ(self, s):
        '''
        Determines the action with max Q value in state s
        '''
        maxQ = self.max_Q(s)
        opt_acts = [a for a in range(self.nA) if self.Q[s, a] == maxQ]
        return random.choice(opt_acts)

    def update_policy(self, s):
        opt_act = self.argmaxQ(s)
        prob_epsilon = lambda action: 1 - self.epsilon if action == opt_act else self.epsilon/(self.nA-1)
        self.policy[s] = [prob_epsilon(a) for a in range(self.nA)]

    def update(self, next_state, reward, done):
        '''
        Agent updates its model.
        TO BE DEFINED BY SUBCLASS
        '''
        pass


class MC(Agent) :
    '''
    Implements a learning rule with Monte Carlo optimization.
    '''

    def __init__(self, parameters:dict):
        super().__init__(parameters)
        self.first_visit = self.parameters['first_visit']
        self.N = np.zeros((self.nS, self.nA))
   
    def restart(self):
        super().restart()
        self.N = np.zeros((self.nS, self.nA))

    def reset(self):
        super().reset()
        self.N = np.zeros((self.nS, self.nA))

    def update(self, next_state, reward, done):
        '''
        Agent updates its model.
        '''
        if not done:
            # Update records
            self.states.append(next_state)
            self.rewards.append(reward)
        else:
            # if reward is None, this comes from maximum number of rounds
            # otherwise, include state and reward
            if reward is not None:
                self.states.append(next_state)
                self.rewards.append(reward)
            T = len(self.rewards) - 1
            G = 0
            for t in range(T - 1, -1, -1):
                reward = self.rewards[t+1]
                G  = self.gamma*G + reward
                state = self.states[t]
                if self.first_visit and state not in self.states[:t]:
                    action = self.actions[t]
                    self.N[state, action] += 1
                    self.Q[state, action] += 1/self.N[state, action] + (G - self.Q[state, action])
            for s in range(self.nS):
                self.update_policy(s)


class SARSA(Agent) :
    '''
    Implements a SARSA learning rule.
    '''

    def __init__(self, parameters:dict):
        super().__init__(parameters)
        self.alpha = self.parameters['alpha']
   
    def update(self, next_state, reward, done):
        '''
        Agent updates its model.
        '''
        # obtain previous state
        state = self.states[-1]
        # obtain previous action
        action = self.actions[-1]
        # Update records
        self.states.append(next_state)
        self.rewards.append(reward)
        # Get next_action
        next_action = self.make_decision()
        if done:
            estimate = reward
        else:
            # Find bootstrap
            estimate = reward + self.gamma * self.Q[next_state, next_action]
        # Obtain delta
        delta = estimate - self.Q[state, action]
        # Update Q value
        self.Q[state, action] += self.alpha * delta
        self.update_policy(state)
