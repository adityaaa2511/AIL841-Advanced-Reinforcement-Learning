import numpy as np
from general_env import TreasureHunt
from tqdm import tqdm
import matplotlib.pyplot as plt


class TabularPolicy:
    """Simple deterministic policy wrapper."""
    def __init__(self, num_states):
        self.a = np.zeros(num_states, dtype=int)

    def get_action(self, s):
        return self.a[s]


def policy_evaluation(env, V, policy, gamma, k):
    """Run exactly k sweeps of synchronous policy evaluation."""
    S = env.num_states
    A = env.num_actions
    T = env.T              # shape = (S, A, S)
    R = env.reward         # shape = (S,)

    for _ in range(k):
        V_new = np.zeros_like(V)

        for s in range(S):
            a = policy.a[s]
            V_new[s] = np.sum(T[s, a] * (R + gamma * V))

        V = V_new

    return V


def policy_improvement(env, V, gamma):
    """Greedy w.r.t current V."""
    S = env.num_states
    A = env.num_actions
    T = env.T
    R = env.reward

    policy = TabularPolicy(S)
    stable = True

    for s in range(S):
        q_vals = np.zeros(A)
        for a in range(A):
            q_vals[a] = np.sum(T[s, a] * (R + gamma * V))

        best_a = np.argmax(q_vals)
        if best_a != policy.a[s]:
            stable = False
        policy.a[s] = best_a

    return policy, stable


def policy_iteration(env, gamma, k, V_init):
    """Perform full policy iteration using k evaluation sweeps."""
    S = env.num_states
    V = V_init.copy()
    policy = TabularPolicy(S)
    max_iterations = 1000

    norms = []

    for _ in range(max_iterations):
        # 1. Policy Evaluation
        V = policy_evaluation(env, V, policy, gamma, k)
        norms.append(np.linalg.norm(V))

        # 2. Policy Improvement
        policy, stable = policy_improvement(env, V, gamma)
        if stable:
            break

        print(f"Policy iteration: V norm = {np.linalg.norm(V)}, policy stable = {stable}")

    return V, policy, norms


# ---------------------------------------------------------------
# Running Q1 experiments
# ---------------------------------------------------------------

def run_q1(locations, seeds=[0,1,2], ks=[1,2,4,8,16], gamma=0.95):
    results = {}
    best_value_norm = -np.inf
    best_policy = None

    for init_type in ["zeros", "minus100"]:
        results[init_type] = {}

        for k in tqdm(ks):
            results[init_type][k] = []

            for seed in tqdm(seeds):
                print(f"Running k={k}, seed={seed}, init={init_type}")

                env = TreasureHunt(locations, n=7, is_testing=False)
                np.random.seed(seed)

                S = env.num_states
                if init_type == "zeros":
                    V0 = np.zeros(S)
                else:
                    V0 = -100*np.ones(S)
                    # terminal (fort) must have V=0
                    fort = env.locations['fort'][0]
                    fort_state = 0
                    # find all states where ship is on fort
                    for s in range(S):
                        ship_loc, _ = env.locations_from_state(s)
                        if ship_loc == fort:
                            V0[s] = 0

                print(f"Initial V norm: {np.linalg.norm(V0)}")

                V, policy, norms = policy_iteration(env, gamma, k, V0)
                results[init_type][k].append(norms)

                # Check if this policy is the best so far
                current_value_norm = np.linalg.norm(V)
                if current_value_norm > best_value_norm:
                    best_value_norm = current_value_norm
                    best_policy = policy

    np.save("Q1_results.npy", results)
    print("Saved Q1_results.npy")

    return results, best_policy

def plot_norms(results, seeds, ks):
    """Plot the 2-norm series for all k and seeds."""
    plt.figure(figsize=(12, 8))

    for seed_idx, seed in enumerate(seeds):
        plt.subplot(1, len(seeds), seed_idx + 1)
        for k in ks:
            norms = results["zeros"][k][seed_idx]  # Get norms for this k and seed
            plt.plot(norms, label=f"k={k}")

        plt.title(f"Seed = {seed}")
        plt.xlabel("Policy Improvement Step")
        plt.ylabel("2-Norm of Value Estimates")
        plt.legend()
        plt.grid()

    plt.tight_layout()
    plt.savefig("Q1_norms_plot.png")
    plt.close()

def generate_gifs(locations, seeds, policy, gamma=0.95):
    """Generate GIFs for one episode using the optimal policy for each seed."""
    for seed in seeds:
        print(f"Generating GIF for seed={seed}...")

        # Initialize the environment
        env = TreasureHunt(locations, n=7, is_testing=True)
        np.random.seed(seed)

        # Reset the environment to the initial state
        env.reset()

        # Generate the GIF for the optimal policy
        gif_path = f"policy_execution_seed_{seed}.gif"
        env.visualize_policy_execution(policy, path=gif_path)

        print(f"Saved GIF for seed={seed} at {gif_path}")

if __name__ == "__main__":

    locations = {
    'ship': [(0, 0)],
    'land': [(2,0), (2,1), (3,1), (0,5), (0,6), (1,5)],
    'fort': [(6, 6)],
    'pirate': [(3,4), (5,3)],
    'treasure': [(3,0), (1,6)]
    }

    seeds = [42, 1, 2]
    ks = [1, 2, 4, 8, 16]
    results, policy = run_q1(locations)

    # Plot the norms for all k and seeds
    plot_norms(results, seeds, ks)

    # Generate GIFs for the optimal policy
    generate_gifs(locations, seeds, policy)
