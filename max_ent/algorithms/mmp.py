###The following implementation is based on the max margin method. Refer to P. Abbeel and A. Y. Ng, “Apprenticeship Learning via Inverse Reinforcement Learning.” for the algorithm below.

import numpy as np
from max_ent.algorithms import rl as RL
from numpy.linalg import norm
import math
import max_ent.gridworld.trajectory as T
import scipy.special

### Input: Feature Matrix (phi), D (demonstration set), discount (gamma)
### Output: Expected Feature Frequencies 
def ef_from_trajectories(features, trajectories, discount):
    n_features = features.shape[-1]

    fe = np.zeros(n_features)

    for t in trajectories:
        time_step = 0
        for s, a, s_ in t.transitions():
            fe += features[s, a, s_, :] * discount ** time_step
            time_step += 1

    return fe / len(trajectories)


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

def _softmax(x1, x2):
    x_max = np.maximum(x1, x2)
    x_min = np.minimum(x1, x2)
    return x_max + np.log(1.0 + np.exp(x_min - x_max))

def value_iteration_new(p_transition, reward, discount, terminal, eps=1e-5):
    """
    Basic value-iteration algorithm to solve the given MDP, except reward function is different.

    Args:
        reward: The reward signal per state as table
            `[state: Integer, action: Integer, state: Integer] -> reward: Float`.
    """
    n_states, _, n_actions = reward.shape

    p_t = np.moveaxis(p_transition, 1, 2) # 

""""
def Qvalue_iteration(T, R, gamma=0.5, n_iters=10):
    nA = R.shape[0]
    nS = T.shape[0]
    Q = np.zeros((nS,nA)) # initially
    for _ in range(n_iters):
        for s in range(nS): # for all states s
            for a in range(nA): # for all actions a
                sum_sp = 0
                for s_ in range(nS): # for all reachable states s'
                    sum_sp += (T[s][a][s_]*(R[s][a][s_] + gamma*max(Q[s_])))
                Q[s][a] = sum_sp
    return Q
"""


def mmp(nominal_rewards, p_transition, features, terminal, trajectories, optim, init, discount,
         eps=1e-4, eps_error=1e-2, burnout=10, max_iter=10000, max_penalty=200, log=None, initial_omega=None):

    n_states, n_actions, _, n_features = features.shape

    # Don't count transitions that start with a terminal state
    features[terminal] = 0

    # Compute expert feature expectation from expert demonstrations
    expert_features = ef_from_trajectories(features, trajectories, discount)
    # Compute probability of a state being the initial start state
    p_initial = initial_probabilities(n_states, n_actions, trajectories)
    # Nominal reward vector is already known
    nominal_rewards = np.array(nominal_rewards)

    ## Variables we will keep track of during iteration
    omega_list = [] # list of weight vectors for each iteration
    omega = np.zeros(n_features) # current weight vector
    if initial_omega is not None:
        omega = initial_omega.copy()
    margin = []
    policy_list = [] # list of policy after each iteration

    nonexpert_feature_expectations = []
    nonexpert_feature_expectations_blended = []

    reward = nominal_rewards - features @ omega

    # First, compute random policy. To do this, we need a reward function to use for value iteration.
    q_function, v_function = RL.value_iteration(p_transition, reward, discount)
    policy = RL.stochastic_policy_from_q_value(q_function) #get policy from running RL with initial reward function
    policy_exec = T.stochastic_policy_adapter(policy)
    policy_list.append(policy)

    for i in range(burnout):

        #get latest policy
        policy = policy_list[-1]

        #Sample new trajectories
        initial = np.zeros(n_states)
        initial[0] = 1.0

        #Compute feature expectations of new policy and add to list
        if i==0:
            sample_tjs = T.generate_trajectories_noworld(1000, n_states, p_transition, policy_exec, initial, terminal)
            nonexpert_feature_expectations.append(ef_from_trajectories(features, sample_tjs, discount))
            print("expert feature expectation: ", expert_features[0:4])
            margin.append(1)
            omega_list.append(omega)
        else:
            if i==1:
                nonexpert_feature_expectations_blended.append(nonexpert_feature_expectations[i-1])
                omega = expert_features - nonexpert_feature_expectations[i-1]
                omega_list.append(omega)
                margin.append(norm((expert_features-nonexpert_feature_expectations_blended[i-1]), 2))
            else:
                A = nonexpert_feature_expectations_blended[i-2]
                B = nonexpert_feature_expectations[i-1]-A
                C = expert_features - nonexpert_feature_expectations_blended[i-2]
                nonexpert_feature_expectations_blended.append(A+(np.dot(B,C)/np.dot(B,B))*(B))
                omega = expert_features-nonexpert_feature_expectations_blended[i-1]
                omega_list.append(omega)
                margin.append(norm((expert_features-nonexpert_feature_expectations_blended[i-1]), 2))

            print("margin: ", margin[-1])
            print("first few weights: ", omega[0:4])
            reward = nominal_rewards - features @ omega
            q_new, v_new = RL.value_iteration(p_transition, reward, discount)
            new_policy = RL.stochastic_policy_from_q_value(q_new)
            new_policy_exec = T.stochastic_policy_adapter(new_policy)
            policy_exec = new_policy_exec
            policy_list.append(new_policy)
            sample_tjs = T.generate_trajectories_noworld(1000, n_states, p_transition, policy_exec, initial, terminal)
            nonexpert_feature_expectations.append(ef_from_trajectories(features, sample_tjs, discount))

    #print(nonexpert_feature_expectations[-1])    
    return omega_list[-1]



### NOTES

#Features shape: (81, 8, 81, 92)  Reward Shape: (81, 8, 81)  Policy shape: (81, 8)
# P_intial shape: (81, 8)  p_transition shape: (81, 81, 8)

# https://github.com/aaronsnoswell/irl_methods/blob/master/irl_methods/projection.py
# https://github.com/rhklite/apprenticeship_inverse_RL/blob/master/Apprenticeship_Inverse_Reinforcement_Learning.ipynb