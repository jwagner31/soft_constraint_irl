###The following implementation is based on the max margin method. Refer to P. Abbeel and A. Y. Ng, “Apprenticeship Learning via Inverse Reinforcement Learning.” for the algorithm below.

import numpy as np
from max_ent.algorithms import rl as RL
from numpy.linalg import norm
import math

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
    margin_list = [] # list of margin for each iteration
    margin = 1 # current margin; dummy margin set to start
    policy_list = [] # list of policy after each iteration

    feature_expectations = []
    feature_expectations_temp = []

    for i in range(burnout):
        print("Iteration: ", i)
        if i==0: #initialize variables
            # First, compute random policy. To do this, we need a reward function to use for value iteration.
            reward = features @ omega # initial reward function
            q_function, v_function = RL.value_iteration(p_transition, reward, discount)
            policy = RL.stochastic_policy_from_q_value(q_function) #get policy from running RL with initial reward function
            policy_list.append(policy)
            #compute ef from this policy - renamed to D to match with paper
            d = forward(p_transition, p_initial, policy, terminal)
            df = (d[:, :, :, None] * features).sum((0, 1, 2))
            print(df.shape)
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
