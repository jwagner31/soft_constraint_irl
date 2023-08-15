###The following implementation is based on the max margin method. Refer to P. Abbeel and A. Y. Ng, “Apprenticeship Learning via Inverse Reinforcement Learning.” for the algorithm below.

import numpy as np
from max_ent.algorithms import rl as RL
from numpy.linalg import norm

### Input: Feature Matrix (phi), D (demonstration set)
### Output: Expected Feature Frequencies 
def ef_from_trajectories(features, trajectories):
    n_features = features.shape[-1]

    fe = np.zeros(n_features)

    for t in trajectories:
        for s, a, s_ in t.transitions():
            fe += features[s, a, s_, :]

    return fe / len(trajectories)


### Input: # of states, # of actions, D (Demonstration set)
### Output: P_start ()
def initial_probabilities(n_states, n_actions, trajectories):
    initial = np.zeros((n_states, n_actions))
    for t in trajectories:
        s, a, _ = list(t.transitions())[0]
        initial[s, a] += 1
    return initial / len(trajectories)


###
def projection_irl(nominal_rewards, p_transition, features, terminal, trajectories, optim, init, discount,
         eps=1e-4, eps_error=1e-2, burnout=10, max_iter=10000, max_penalty=200, log=None, initial_omega=None):

    n_states, n_actions, _, n_features = features.shape

    # Don't count transitions that start with a terminal state
    features[terminal] = 0

    # Compute expert feature expectation from expe rt demonstrations
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

    feature_expectations = []
    feature_expectations_temp = []

    for i in range(burnout):
        print("Iteration: ", i)
        if i==0:
            # First, compute random policy. To do this, we need a reward function to use for value iteration.
            reward = nominal_rewards - features @ omega # initial reward function
            q_function, v_function = RL.value_iteration(p_transition, reward, discount)
            policy = RL.stochastic_policy_from_q_value(world, q_function, omega) #confused on this step
            #compute ef from this policy
            feature_expectations.append()
            omega_list.append(omega)
            margin_list.append(margin)
        else:
            if i==1:
                feature_expectations_temp.append(feature_expectations[i-1])
                omega_list.append(expert_features-feature_expectations[i-1])
                margin.append(norm((expert_features-feature_expectations_temp[i-1]),2))



def _softmax(x1, x2):
    x_max = np.maximum(x1, x2)
    x_min = np.minimum(x1, x2)
    return x_max + np.log(1.0 + np.exp(x_min - x_max))
