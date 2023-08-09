###The following implementation is based on the max margin method. Refer to P. Abbeel and A. Y. Ng, “Apprenticeship Learning via Inverse Reinforcement Learning.” for the algorithm below.

import numpy as np

# Input: feat
def ef_from_trajectories(features, trajectories):
    n_features = features.shape[-1]

    fe = np.zeros(n_features)

    for t in trajectories:
        for s, a, s_ in t.transitions():
            fe += features[s, a, s_, :]

    return fe / len(trajectories)

# Input: # of states, # of actions, D (Demonstration set)
# Output: P_start ()
def initial_probabilities(n_states, n_actions, trajectories):
    initial = np.zeros((n_states, n_actions))
    for t in trajectories:
        s, a, _ = list(t.transitions())[0]
        initial[s, a] += 1
    return initial / len(trajectories)



def mmp(nominal_rewards, p_transition, features, terminal, trajectories, optim, init, discount,
         eps=1e-4, eps_error=1e-2, burnout=100, max_iter=10000, max_penalty=200, log=None, initial_omega=None):

    n_states, n_actions, _, n_features = features.shape

    # Don't count transitions that start with a terminal state
    features[terminal] = 0

    # compute static properties from trajectories
    expert_features = ef_from_trajectories(features, trajectories)
    p_initial = initial_probabilities(n_states, n_actions, trajectories)
    nominal_rewards = np.array(nominal_rewards)

    omega = init(n_features)
    if initial_omega is not None:
        omega = initial_omega.copy()
    delta = mean_error = np.inf

    #optim.reset(omega)  --> replace with our choice of optimizer
    epoch = 0
    best = None
    best_error = 100000
    while epoch <= burnout or (delta > eps and mean_error > eps_error and epoch < max_iter):
        omega_old = omega.copy()

        # compute per-state reward
        reward = nominal_rewards - features @ omega

        # Backward, Forward
        # Takes in reward and spits out initial policy. We want to to do this in style of mmp rather than max ent inference procedure
        policy = backward_causal(p_transition, reward, terminal, discount) 
        d = forward(p_transition, p_initial, policy, terminal)

        # compute the gradient
        # df[i] is the expected visitation for feature i accross all (s, a, s_)
        # df[i] = [d[s, a, s_, i] * features[s, a, s_, i] for all (s, a, s_)].sum()
        df = (d[:, :, :, None] * features).sum((0, 1, 2))
        grad = df - expert_features

        mean_error = np.abs(grad).mean()

        if epoch >= burnout and mean_error < best_error:
            best = omega.copy()
            best_error = mean_error

        # perform optimization step and compute delta for convergence
        optim.step(grad)

        if omega.max() > max_penalty:
            omega = omega * (max_penalty / omega.max())
            optim.reset(omega)

        delta = np.max(np.abs(omega_old - omega))

        if log is not None and type(log) == list:
            log.append({
                'omega': omega_old.copy(),
                'delta': delta,
                'epoch': epoch,
                'mean_error': mean_error,
                'is_best': mean_error == best_error,
                'best_omega': omega_old.copy() if best is None else best.copy(),
                'best_reward': reward
            })

        if epoch % 100 == 0:
            print(f'MAE(best): {min(mean_error, best_error): 0.15f}')
        epoch += 1

    print(f'Finished with MAE(best): {best_error: 0.15f}')

    return best if best is not None else omega


def _softmax(x1, x2):
    x_max = np.maximum(x1, x2)
    x_min = np.minimum(x1, x2)
    return x_max + np.log(1.0 + np.exp(x_min - x_max))
