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

#Features shape: (81, 8, 81, 92)  Reward Shape: (81, 8, 81)  Policy shape: (81, 8)
# P_intial shape: (81, 8)  p_transition shape: (81, 81, 8)

# https://github.com/aaronsnoswell/irl_methods/blob/master/irl_methods/projection.py
# https://github.com/rhklite/apprenticeship_inverse_RL/blob/master/Apprenticeship_Inverse_Reinforcement_Learning.ipynb


def mmp(nominal_rewards, p_transition, features, terminal, trajectories, optim, init, discount,
         eps=1e-4, eps_error=1e-2, burnout=30, max_iter=10000, max_penalty=200, log=None, initial_omega=None):

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
    omega = init(n_features) # current weight vector
    if initial_omega is not None:
        omega = initial_omega.copy()
    policy_list = [] # list of policy after each iteration

    nonexpert_feature_expectations = np.zeros(shape=(0, n_features))
    nonexpert_feature_expectations_blended = np.zeros(shape=(0, n_features))

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
        sample_tjs = T.generate_trajectories(1000, n_states, policy_exec, initial, terminal)

        #Compute feature expectations of new policy and add to list
        nonexpert_feature_expectations = np.vstack(
                (
                    nonexpert_feature_expectations,
                    nonexpert_feature_expectations(features, sample_tjs, discount)))
        

        mu_e = expert_features 
        mu_prev = nonexpert_feature_expectations[-1]

        mu_prev_prev = nonexpert_feature_expectations[-2]
        mu_bar_prev_prev = nonexpert_blended_feature_expectations[-2, :]

        # The below finds the orthogonal projection of the expert's
        # feature expectations onto the line through mu_prev and
        # mu_prev_prev
        mu_bar_prev = mu_bar_prev_prev \
            + (mu_prev - mu_bar_prev_prev).T \
                @ (mu_e - mu_bar_prev_prev) \
            / (mu_prev - mu_bar_prev_prev).T \
                @ (mu_prev - mu_bar_prev_prev) \
            * (mu_prev - mu_bar_prev_prev)

        nonexpert_blended_feature_expectations = np.vstack(
            (
                nonexpert_blended_feature_expectations,
                mu_bar_prev
            )
        )

        omega = mu_e - mu_bar_prev
        omega_list.append(omega)

        reward = nominal_rewards - features @ omega
        error = np.linalg.norm(w)
        print(error)

        q_new, v_new = RL.value_iteration(p_transition, reward, discount)
        new_policy = RL.stochastic_policy_from_q_value(q_new)
        new_policy_exec = T.stochastic_policy_adapter(new_policy)
        policy_list.append(new_policy)

    return omega_list[-1]