import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import random
from copy import deepcopy
from general_env import TreasureHunt


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# Utilities
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def state_to_feature(state):
    """
    Convert state to feature vector.
    State assumed to be integer index (0-195).
    """
    one_hot = np.zeros(196)
    one_hot[state] = 1.0
    return one_hot


def epsilon_greedy(Q_values, epsilon=0.1):
    if np.random.rand() < epsilon:
        return np.random.randint(len(Q_values))
    return np.argmax(Q_values)


# ============================================================
# Linear Q Approximator
# ============================================================

class LinearQ:
    def __init__(self, num_states=196, num_actions=4, lr=1e-2):
        self.num_actions = num_actions
        self.W = np.zeros((num_actions, num_states))
        self.lr = lr

    def predict(self, state):
        phi = state_to_feature(state)
        return self.W @ phi

    def update(self, dataset):
        for s, a, target in dataset:
            phi = state_to_feature(s)
            pred = self.W[a] @ phi
            grad = (pred - target) * phi
            self.W[a] -= self.lr * grad


# ============================================================
# Neural Network Q Approximator
# ============================================================

class QNetwork(nn.Module):
    def __init__(self, input_dim=196, output_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class NeuralQ:
    def __init__(self, lr=1e-3):
        self.model = QNetwork().to(DEVICE)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def predict(self, state):
        with torch.no_grad():
            x = torch.FloatTensor(state_to_feature(state)).to(DEVICE)
            return self.model(x).cpu().numpy()

    def update(self, dataset):

        states = []
        actions = []
        targets = []

        for s, a, target in dataset:
            states.append(state_to_feature(s))
            actions.append(a)
            targets.append(target)

        states = np.array(states, dtype=np.float32)
        targets = np.array(targets, dtype=np.float32)

        states = torch.from_numpy(states).to(DEVICE)
        actions = torch.tensor(actions, dtype=torch.long, device=DEVICE)
        targets = torch.tensor(targets, dtype=torch.float32, device=DEVICE)

        pred_all = self.model(states)
        pred = pred_all.gather(1, actions.unsqueeze(1)).squeeze()

        loss = self.loss_fn(pred, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# ============================================================
# Neural Network with Target Network
# ============================================================

class NeuralQTarget:
    def __init__(self, lr=1e-3):
        self.model = QNetwork().to(DEVICE)
        self.target_model = deepcopy(self.model)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def predict(self, state):
        with torch.no_grad():
            x = torch.FloatTensor(state_to_feature(state)).to(DEVICE)
            return self.model(x).cpu().numpy()

    def predict_target(self, state):
        with torch.no_grad():
            x = torch.FloatTensor(state_to_feature(state)).to(DEVICE)
            return self.target_model(x).cpu().numpy()

    def update(self, dataset):

        states = []
        actions = []
        targets = []

        for s, a, target in dataset:
            states.append(state_to_feature(s))
            actions.append(a)
            targets.append(target)

        # Convert to numpy first (fast)
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int64)
        targets = np.array(targets, dtype=np.float32)

        # Convert to torch tensors
        states = torch.from_numpy(states).to(DEVICE)
        actions = torch.from_numpy(actions).to(DEVICE)
        targets = torch.from_numpy(targets).to(DEVICE)

        # Forward pass
        pred_all = self.model(states)

        # Select Q(s,a) for each sample
        pred = pred_all.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute loss
        loss = self.loss_fn(pred, targets)

        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ===== Soft target update =====
        tau = 0.05
        with torch.no_grad():
            for target_param, param in zip(self.target_model.parameters(),
                                        self.model.parameters()):
                target_param.data.copy_(
                    (1 - tau) * target_param.data + tau * param.data
                )


# ============================================================
# Approximate Policy Iteration
# ============================================================

def collect_data(env, Q, gamma):
    dataset = []

    for _ in range(200):
        state = env.reset()

        # roll-in
        steps = np.random.randint(1, 5)
        for _ in range(steps):
            action = np.argmax(Q.predict(state))
            state, _, done, _ = env.step(action)
            if done:
                break

        # evaluation
        action = np.random.randint(4)
        next_state, reward, done, _ = env.step(action)

        current_state = next_state

        # roll-out
        total_return = reward
        discount = gamma
        while not done:
            # Use target network if available
            if hasattr(Q, "predict_target"):
                next_action = np.argmax(Q.predict_target(current_state))
            else:
                next_action = np.argmax(Q.predict(current_state))

            current_state, r, done, _ = env.step(next_action)

            total_return += discount * r
            discount *= gamma

        dataset.append((state, action, total_return))

    return dataset


def test_policy(env, Q):
    total_reward = 0
    treasures = 0

    for _ in range(5):
        state = env.reset()
        done = False

        while not done:
            action = np.argmax(Q.predict(state))
            state, reward, done, _ = env.step(action)
            total_reward += reward
            treasures += 1 if reward == 2 else 0

    return total_reward / 5, treasures / 5


def run_api(approximator_class, seed, locations=None):
    set_seed(seed)

    env = TreasureHunt(locations=locations, n=7, is_testing=False)
    Q = approximator_class()

    gamma = 0.75
    N = 750

    avg_targets = []
    test_rewards = []
    value_norms = []
    treasure_counts = []

    for i in range(N):
        dataset = collect_data(env, Q, gamma)
        avg_targets.append(np.mean([d[2] for d in dataset]))

        Q.update(dataset)

        # compute V norm
        V = []
        for s in range(196):
            V.append(np.max(Q.predict(s)))
        value_norms.append(np.linalg.norm(V))

        # test
        test_env = TreasureHunt(locations=locations,n=7, is_testing=True)
        reward, treasures = test_policy(test_env, Q)

        test_rewards.append(reward)
        treasure_counts.append(treasures)

        print(f"Iter {i} | Reward {reward:.2f}")

    return avg_targets, test_rewards, value_norms, treasure_counts


# ============================================================
# Plotting
# ============================================================

def plot_results(results, title, seed):
    avg_targets, test_rewards, value_norms, treasure_counts = results

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    axs[0, 0].plot(avg_targets)
    axs[0, 0].set_title("Avg Q Target")

    axs[0, 1].plot(test_rewards)
    axs[0, 1].set_title("Mean Test Reward")

    axs[1, 0].plot(value_norms)
    axs[1, 0].set_title("||V||_2")

    axs[1, 1].plot(treasure_counts)
    axs[1, 1].set_title("Treasures Collected")

    fig.suptitle(title + f" | Seed {seed}")
    plt.tight_layout()
    plt.savefig(f"{title}_{seed}_results.png")
    plt.close()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    locations = {
        'ship': [(0, 0)],
        'land': [
            (2, 0), (2, 1), (3, 1), 
            (0, 5), (0, 6), (1, 5)  
        ],
        'fort': [(6, 6)],
        'pirate': [(3, 4), (5, 3)], 
        'treasure': [(3, 0), (1, 6)] 
    }

    seeds = [0, 1, 2, 42]

    for seed in seeds:
        # print(f"\n==== Linear | Seed {seed} ====")
        # results = run_api(LinearQ, seed, locations)
        # plot_results(results, "Linear Q",seed)

        print(f"\n==== Neural Net | Seed {seed} ====")
        results = run_api(NeuralQ, seed, locations)
        plot_results(results, "Neural Network Q",seed)

        print(f"\n==== Neural Net + Target | Seed {seed} ====")
        results = run_api(NeuralQTarget, seed, locations)
        plot_results(results, "Neural Network + Target Q",seed)