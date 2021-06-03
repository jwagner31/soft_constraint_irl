#!/usr/bin/env python

from max_ent.gridworld.gridworld import Directions
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
Demonstration = namedtuple('Demonstration', ['trajectories', 'policy'])
ICRL_Result = namedtuple('ICRL_Result', ['omega', 'reward',
                                         'state_weights', 'action_weights', 'color_weights'])


def setup_mdp(size, feature_list, constraints,
              terminal=[20], start=[0], terminal_reward=10, p_slip=0.1):
    # create our world
    world = W.IcyGridWorld(size=size, feature_list=feature_list,
                           allow_diagonal_actions=True, p_slip=p_slip)
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

    return MDP(world, reward, terminal, start)


def generate_trajectories(world, reward, start, terminal, n_trajectories=200):
    """
    Generate some "expert" trajectories.
    """
    # parameters
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

    return Demonstration(tjs, policy)


def generate_weighted_average_trajectories(world, n_r, c_r, start, terminal, weights):

    # parameters
    n_trajectories = 200
    discount = 0.9

    # set up initial probabilities for trajectory generation
    initial = np.zeros(world.n_states)
    initial[start] = 1.0

    # generate trajectories
    q_n, _ = RL.value_iteration(world.p_transition, n_r, discount)
    q_c, _ = RL.value_iteration(world.p_transition, c_r, discount)
    avg_q = q_n * weights[0] + q_c * weights[1]

    policy = RL.stochastic_policy_from_q_value(world, avg_q)
    policy_exec = T.stochastic_policy_adapter(policy)
    tjs = list(T.generate_trajectories(n_trajectories,
                                       world, policy_exec, initial, terminal))
    return Demonstration(tjs, None)


def generate_mdft_trajectories(world, n_r, c_r, start, terminal, w):

    # parameters
    n_trajectories = 200
    discount = 0.9

    # set up initial probabilities for trajectory generation
    initial = np.zeros(world.n_states)
    initial[start] = 1.0

    # generate trajectories
    q_n, _ = RL.value_iteration(world.p_transition, n_r, discount)
    q_c, _ = RL.value_iteration(world.p_transition, c_r, discount)
    policy_exec = T.mdft_policy_adapter(q_n, q_c, w=np.array(w))
    tjs = list(T.generate_trajectories(n_trajectories,
                                       world, policy_exec, initial, terminal))

    return Demonstration(tjs, None)


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


def convert_constraints_to_probs(nominal_reward, learned_params):
    std_n = nominal_reward.std()
    std_l = learned_params.reward.std()
    std_pooled = np.sqrt((std_n**2 + std_l**2)/2)

    def convert_to_probs(w):
        w = -w.copy()  # reward -> penalty
        w = (w - std_pooled) / std_pooled
        return 1 / (1 + np.exp(-w))

    w_s = learned_params.state_weights
    p_s = convert_to_probs(w_s)

    keys = list(learned_params.action_weights.keys())
    w_a = np.array([learned_params.action_weights[a] for a in keys])
    p_a = convert_to_probs(w_a)
    p_a = {keys[i]: p_a[i] for i in range(8)}

    w_c = learned_params.color_weights
    p_c = convert_to_probs(w_c)

    return p_s, p_a, p_c

def convert_constraints_to_probs2(n_cfg, learned_params):
    pen = n_cfg.mdp.world.phi @ learned_params.omega
    std_n = n_cfg.mdp.reward.std()
    std_l = learned_params.reward.std()
    std_pooled = np.sqrt((std_n**2 + std_l**2)/2)    
    pen = (pen - std_pooled) / std_pooled
    p = 1 / (1 + np.exp(-pen))
    p_s = p.mean((0, 1))
    p_a = p.mean((0, 2))
    p_c = np.array([0, p[:, :, n_cfg.blue].mean(), p[:, :, n_cfg.green].mean()])
    p_s[n_cfg.blue] = np.minimum(1 - p_c[1], p_s[n_cfg.blue])
    p_s[n_cfg.green] = np.minimum(1 - p_c[2], p_s[n_cfg.green])
    p_s[n_cfg.mdp.start] = 0
    p_s[n_cfg.mdp.terminal] = 0
    p_a = {Directions.ALL_DIRECTIONS[i]: p_a[i] for i in range(8)}
    return p_s, p_a, p_c    
