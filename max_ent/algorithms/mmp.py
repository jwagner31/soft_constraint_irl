###The following implementation is based on the max margin method. Refer to P. Abbeel and A. Y. Ng, “Apprenticeship Learning via Inverse Reinforcement Learning.” for the algorithm below.

import numpy as np
from max_ent.algorithms import rl as RL
from numpy.linalg import norm
import math
import max_ent.gridworld.trajectory as T
import max_ent.optim as O
import scipy.special
import max_ent.algorithms.mmp_helper as H
import max_ent.algorithms.robsvm as R
from sklearn.svm import SVC


def initialization():


    return True

#Features shape: (81, 8, 81, 92)  Reward Shape: (81, 8, 81)  Policy shape: (81, 8)
# P_intial shape: (81, 8)  p_transition shape: (81, 81, 8)


def mmp(world, nominal_rewards, p_transition, features, terminal, trajectories, optim, init, discount,
         eps=1e-4, eps_error=1e-2, burnout=50, max_iter=10000, max_penalty=200, log=None, initial_omega=None, method="projection"):

    ### INITIALIZE VARIABLES
    n_states, n_actions, _, n_features = features.shape
    features[terminal] = 0  # Don't count transitions that start with a terminal state
    p_initial = H.initial_probabilities(n_states, n_actions, trajectories) # Compute probability of a state being the initial start state
    nominal_rewards = np.array(nominal_rewards) # Nominal reward vector is already known
    initial = np.zeros(world.n_states)
    initial[0] = 1.0

    ### TRACKING VARIABLES
    policy_list  = []
    margin = []
    fe_current = []
    fe_bar = []
    omega_c = []

    fe_expert = H.fe_from_trajectories(features, trajectories, discount)
    #print(fe_expert)

    ### Step 3 of MESC-IRL
    omega_r = np.zeros(n_features)
    reward_c = nominal_rewards - features @ omega_r
    reward_c[:,:,:] = 0

    # First, compute random policy. To do this, we need a reward function to use for value iteration.
    q_function, v_function = H.value_iteration(p_transition, reward_c, discount)
    policy = H.stochastic_policy_from_q_value(q_function) #get policy from running RL with initial reward function
    policy_exec = H.stochastic_policy_adapter(policy)
    policy_list.append(policy)
    sample_tjs = None

    for i in range(burnout):
        print("Iteration: ", i)
        if method == "projection":
            #get latest policy
            policy = policy_list[-1]

            #Compute feature expectations of new policy and add to list
            if i==0:
                sample_tjs = T.generate_trajectories(1000, world, policy_exec, initial, terminal) #get sample trajectories from initial policy
                fe_current.append(H.fe_from_trajectories(features, sample_tjs, discount)) # get feature expectation of initial policy using sampled trajectories
                margin.append(1) # dummy margin to start
                omega_c.append(np.zeros(n_features)) # dummy omega_c to start
            else:
                if i==1:
                    fe_bar.append(fe_current[i-1])
                    omega_c.append(fe_expert - fe_current[i-1])
                    margin.append(norm((fe_expert-fe_bar[i-1]), 2))
                else:
                    A = fe_bar[i-2]
                    B = fe_current[i-1]-A
                    C = fe_expert - fe_bar[i-2]
                    fe_bar.append(A+(np.dot(B,C)/np.dot(B,B))*(B))
                    omega_c.append(fe_expert - fe_bar[i-1])
                    margin.append(norm((fe_expert-fe_bar[i-1]), 2))
                    print("margin: ", margin[i])

                reward_c = features @ omega_c[i]
                #print("reward_c: ", reward_c[1,1,:])
                q_new, v_new = H.value_iteration(p_transition, reward_c, discount)
                new_policy = H.stochastic_policy_from_q_value(q_new)  # (s, a) -> p(a | s)
                new_policy_exec = H.stochastic_policy_adapter(new_policy)
                policy_exec = new_policy_exec
                policy_list.append(new_policy)
                sample_tjs = T.generate_trajectories(1000, world, policy_exec, initial, terminal)
                fe_current.append(H.fe_from_trajectories(features, sample_tjs, discount))
        else: 
            nonexpert_feature_expectations = np.zeros(shape = (0, n_features))

            #get latest policy
            policy = policy_list[-1]
            policy_exec = H.stochastic_policy_adapter(policy)
            sample_tjs = T.generate_trajectories(1000, world, policy_exec, initial, terminal)
            nonexpert_feature_expectations = np.vstack((nonexpert_feature_expectations, H.fe_from_trajectories(features, sample_tjs, discount)))

            X = np.vstack((fe_expert, nonexpert_feature_expectations))
            y = np.array([1] + [-1] * len(nonexpert_feature_expectations))
            # Train a standard SVM
            clf = SVC(kernel='linear')
            clf.fit(X, y)

            # Get the weight vector w and intercept b from the trained SVM
            w = clf.coef_[0]
            b = clf.intercept_[0]

            print(np.linalg.norm(w))

            reward_c = features @ w
            q_new, v_new = H.value_iteration(p_transition, reward_c, discount)
            new_policy = H.stochastic_policy_from_q_value(q_new)
            policy_list.append(new_policy)

    #print(nonexpert_feature_expectations[-1])
    return reward_c, sample_tjs




"""
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


"""