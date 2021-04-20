#!/usr/bin/env python

import max_ent.gridworld as W
from max_ent.algorithms import rl as RL
from max_ent.algorithms import icrl as ICRL
import max_ent.gridworld.trajectory as T
import max_ent.optim as O
import max_ent.gridworld.feature as F

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import math
from collections import namedtuple, defaultdict
from logging import debug, root, DEBUG


MDP = namedtuple('MDP', ['world', 'reward', 'terminal', 'start'])
Demonstraition = namedtuple('Demonstraition', ['trajectories', 'policy'])
ICRL_Result = namedtuple('ICRL_Result', ['omega', 'reward',
                                         'state_weights', 'action_weights', 'color_weights'])


def setup_mdp(size, feature_list, constraints,
              terminal=[20], terminal_reward=10, default_reward=-0.01):
    # create our world
    world = W.IcyGridWorld(size=size, feature_list=feature_list,
                           allow_diagonal_actions=True, p_slip=0.1)
    # set up the reward function
    reward = np.zeros((world.n_states, world.n_actions, world.n_states))
    reward[:, :, terminal] = terminal_reward

    offset = {}
    o = 0
    for f in feature_list:
        offset[f] = o
        o += f.size

    for f, v, r in constraints:
        idx = world.phi[:, :, :, offset[f]: offset[f] + f.size] == f.value2feature(v)
        idx = idx.all(-1)
        reward[idx] += r

    reward[reward == 0] = default_reward

    return MDP(world, reward, terminal, [0])


def generate_trajectories(world, reward, start, terminal):
    """
    Generate some "expert" trajectories.
    """
    # parameters
    n_trajectories = 200
    discount = 0.9

    # set up initial probabilities for trajectory generation
    initial = np.zeros(world.n_states)
    initial[start] = 1.0

    # generate trajectories
    q, _ = RL.value_iteration(world.p_transition, reward, discount)
    policy = RL.stochastic_policy_from_q_value(world, q)
    policy_exec = T.stochastic_policy_adapter(policy)
    tjs = list(T.generate_trajectories(n_trajectories,
                                       world, policy_exec, initial, terminal))

    return Demonstraition(tjs, policy)


def generate_mdft_trajectories(world, n_r, c_r, start, terminal):

    # parameters
    n_trajectories = 200
    discount = 0.9

    # set up initial probabilities for trajectory generation
    initial = np.zeros(world.n_states)
    initial[start] = 1.0

    # generate trajectories
    q_n, _ = RL.value_iteration(world.p_transition, n_r, discount)
    q_c, _ = RL.value_iteration(world.p_transition, c_r, discount)
    policy_exec = T.mdft_policy_adapter(q_n, q_c, w=np.array([0.5, 0.5]))
    tjs = list(T.generate_trajectories(n_trajectories,
                                       world, policy_exec, initial, terminal))

    return Demonstraition(tjs, None)


def learn_constraints(nominal_rewards, world, terminal, trajectories, discount=0.9):
    init = O.Constant(1e-6)
    optim = O.ExpSga(lr=0.3, clip_grad_at=10)

    omega = ICRL.icrl(nominal_rewards, world.p_transition, world.phi,
                      terminal, trajectories, optim, init, discount, max_iter=500)

    reward = nominal_rewards - world.phi @ omega
    omega_action = {a: -omega[world.n_states + i]
                    for i, a in enumerate(world.actions)}
    omega_state = -omega[:world.n_states]
    omage_color = -omega[world.n_states + world.n_actions:]

    return ICRL_Result(omega, reward, omega_state, omega_action, omage_color)
