###The following implementation is based on the max margin method. Refer to P. Abbeel and A. Y. Ng, “Apprenticeship Learning via Inverse Reinforcement Learning.” for the algorithm below.

import numpy as np
from max_ent.algorithms import rl as RL
from numpy.linalg import norm
import math
import max_ent.gridworld.trajectory as T

def backward_causal(p_transition, reward, terminal, discount, eps=1e-5):
    n_states, _, n_actions = p_transition.shape

    # set up terminal reward function
    if len(terminal) == n_states:
        reward_terminal = np.array(terminal, dtype=np.float)
    else:
        reward_terminal = -np.inf * np.ones(n_states)
        reward_terminal[terminal] = 0.0

    # compute state log partition V and state-action log partition Q
    v = -1e200 * np.ones(n_states)  # np.dot doesn't behave with -np.inf

    p_t = discount * np.moveaxis(p_transition, 1, 2)
    r = (reward * p_t).sum(-1)  # computes the state-action rewards

    delta = np.inf

    while delta > eps:
        v_old = v
        q = r + p_t @ v_old

        v = reward_terminal
        for a in range(n_actions):
            v = _softmax(v, q[:, a])

        delta = np.max(np.abs(v - v_old))

    # compute and return policy
    return np.exp(q - v[:, None])


def forward(p_transition, p_initial, policy, terminal, eps=1e-5):
    # Don't allow transitions from terminal
    p_t = np.moveaxis(p_transition.copy(), 1, 2)
    p_terminal = p_t[terminal, :, :].copy()
    p_t[terminal, :, :] = 0.0

    d = p_initial.sum(1)
    d_total = d

    delta = np.inf
    while delta > eps:
        # state-action expected visitation
        d_sa = d[:, None] * policy
        # for each state s, multiply the expected visitations of all states to s by their probabilities
        # d_s = sum(sa_ev[s_from, a] * p_t[s_from, a, s] for all s_from, a)
        d_ = (d_sa[:, :, None] * p_t).sum((0, 1))

        delta = np.max(np.abs(d - d_))
        d = d_
        d_total += d

    p_t[terminal, :, :] = p_terminal
    # Distribute the visitation stats of satate to their actions
    d_sa = d_total[:, None] * policy

    # Distribute state-action visitations to the next states
    d_transition = d_sa[:, :, None] * p_t

    return d_transition



### Input: Feature Matrix (phi), D (demonstration set)
### Output: Expected Feature Frequencies 
def ef_from_trajectories(features, trajectories):
    n_features = features.shape[-1]

    fe = np.zeros(n_features)

    for t in trajectories:
        for s, a, s_ in t.transitions():
            fe += features[s, a, s_, :]

    return fe / len(trajectories)

def feature_expectation_from_policy(features, policy):
    n_features = features.shape[-1]
    n_states, n_actions, _, _ = features.shape

    fe = np.zeros(n_features)

    for s in range(n_states):
        for a in range(n_actions):
            fe += policy[s, a] * features[s, a, :, :]

    return fe


def sigmoid(arry):
    sig=[]
    for i in arry:
        sig.append(1/(1+math.exp(-i)))           
    return np.array(sig)


### Input: # of states, # of actions, D (Demonstration set)
### Output: P_start ()
def initial_probabilities(n_states, n_actions, trajectories):
    initial = np.zeros((n_states, n_actions))
    for t in trajectories:
        s, a, _ = list(t.transitions())[0]
        initial[s, a] += 1
    return initial / len(trajectories)

def sample_trajectories_from_policy(n_states, policy, start, terminal, n_trajectories=200):
    # set up initial probabilities for trajectory generation
    initial = np.zeros(n_states)
    initial[start] = 1.0

    # generate trajectories
    policy_exec = T.stochastic_policy_adapter(policy)
    tjs = list(T.generate_trajectories(n_trajectories,
                                       world, policy_exec, initial, terminal))
    return Demonstration(tjs, None)


#Features shape: (81, 8, 81, 92)  Reward Shape: (81, 8, 81)  Policy shape: (81, 8)
# P_intial shape: (81, 8)  p_transition shape: (81, 81, 8)

# https://github.com/aaronsnoswell/irl_methods/blob/master/irl_methods/projection.py
# https://github.com/rhklite/apprenticeship_inverse_RL/blob/master/Apprenticeship_Inverse_Reinforcement_Learning.ipynb


###
def mmp(nominal_rewards, p_transition, features, terminal, trajectories, optim, init, discount,
         eps=1e-4, eps_error=1e-2, burnout=30, max_iter=10000, max_penalty=200, log=None, initial_omega=None):

    n_states, n_actions, _, n_features = features.shape

    # Don't count transitions that start with a terminal state
    features[terminal] = 0

    # Compute expert feature expectation from expert demonstrations
    expert_features = ef_from_trajectories(features, trajectories)
    # Compute probability of a state being the initial start state
    p_initial = initial_probabilities(n_states, n_actions, trajectories)
    # Nominal reward vector is already known
    nominal_rewards = np.array(nominal_rewards)

    ## Variables we will keep track of during iteration
    omega_list = [] # list of weight vectors for each iteration
    omega = init(n_features) # current weight vector
    if initial_omega is not None:
        omega = initial_omega.copy()
    delta = mean_error = np.inf 
    optim.reset(omega)

    margin_list = [] # list of margin for each iteration
    margin = 1 # current margin; dummy margin set to start
    policy_list = [] # list of policy after each iteration

    feature_expectations = []
    feature_expectations_temp = []

    reward = nominal_rewards - features @ omega

    for i in range(burnout):
        print("Iteration: ", i)
        if i==0: #initialize variables
            # First, compute random policy. To do this, we need a reward function to use for value iteration.
            q_function, v_function = RL.value_iteration(p_transition, reward, discount)
            policy = RL.stochastic_policy_from_q_value(q_function) #get policy from running RL with initial reward function
            policy_exec = T.stochastic_policy_adapter(policy)
            policy_list.append(policy)
            #compute ef from this policy - renamed to D to match with paper
            d = forward(p_transition, p_initial, policy, terminal)
            df = (d[:, :, :, None] * features).sum((0, 1, 2))
            feature_expectations.append(df)
            omega_list.append(omega)
            margin_list.append(margin)
        else:
            if i==1:
                feature_expectations_temp.append(feature_expectations[i-1])
                omega = expert_features - feature_expectations_temp[i-1]
                omega_list.append(omega)
                margin = norm((expert_features-feature_expectations_temp[i-1]),2)
                margin_list.append(margin)

                print("margin: ", margin_list[i])
            else:
                A = feature_expectations_temp[i-2]
                B=feature_expectations[i-1]-A
                C=expert_features-feature_expectations_temp[i-2]
                feature_expectations_temp.append(A+(np.dot(B,C)/np.dot(B,B))*(B))

                omega = expert_features - feature_expectations_temp[i-1]
                omega_list.append(omega)
                margin = norm((expert_features-feature_expectations_temp[i-1]),2)
                margin_list.append(margin)

                print("margin: ",margin_list[i]) 
            
            #if(margin <= eps):
             #   break
            
            # Compute opimal policy using RL, then use that policy to get feature expectation
            reward = nominal_rewards - features @ omega
            q_function, v_function = RL.value_iteration(p_transition, reward, discount)
            policy = RL.stochastic_policy_from_q_value(q_function) #confused on this step
            policy_list.append(policy)
            d = forward(p_transition, p_initial, policy, terminal)
            df = (d[:, :, :, None] * features).sum((0, 1, 2))
            feature_expectations.append(df)

            mean_error = np.abs(df - expert_features).mean()
            print(mean_error)
            delta = np.max(np.abs(omega_list[i-1] - omega))
            #if(delta <= eps):
              #  break
    return omega_list[-1]




def _softmax(x1, x2):
    x_max = np.maximum(x1, x2)
    x_min = np.minimum(x1, x2)
    return x_max + np.log(1.0 + np.exp(x_min - x_max))
