import numpy as np
import math

class DP_LSVI_Luyo:
    def __init__(self, 
                 H=10, 
                 K=100,  # Number of episodes
                 num_actions=100, 
                 delta=0.01, 
                 eps=10.0,
                 p=0.05, 
                 seed=0, 
                 action_seed=42):
        """
        Initialize the DP-LSVI-Luyo Algorithm with default parameters.

        Environment Setup:
        - model_name (str): Name of the model.
        - H (int): Horizon length (number of stages per episode).
        - K (int): Number of episodes.
        - d (int): Dimension of the feature space.
        - num_actions (int): Number of possible actions.
        - delta (float): Confidence parameter.
        - eps (float): Privacy parameter (ε).
        - p (float): Failure probability.
        - a_vectors (numpy.ndarray): Generated action vectors.
        - alpha_seq (list): Sequence of alpha parameters for transition probabilities.

        Parameters from Appendix E.1:
        - lambda_ (float): Regularization parameter.
        - B (int): Batch size calculated based on provided formulas.
        - B0 (int): Auxiliary parameter based on B.
        - sigma_Lambda (float): Scale parameter for Gaussian noise in Λ matrices.
        - sigma_u (float): Scale parameter for Gaussian noise in η vectors.
        - YJ (float): Auxiliary parameter for parameter updates.
        - cK (float): Auxiliary constant for parameter updates.
        - CJ (float): Auxiliary constant for parameter updates.
        - UK (float): Auxiliary parameter for parameter updates.
        - chi (float): Auxiliary parameter used in beta computation.
        - beta (float): Confidence bound parameter.

        Initialization:
        - Lambda_tilde (list of numpy.ndarray): List of regularization matrices for each stage.
        - w_tilde (list of numpy.ndarray): List of weight vectors for each stage.
        - H_trees (list of list of numpy.ndarray): List of trees containing symmetric Gaussian matrices.
        - eta (numpy.ndarray): Array of Gaussian noise vectors.
        - k_indices (list of int): List of episode indices where batch updates occur.
        - all_states (numpy.ndarray): Array to store states for all episodes and stages.
        - all_actions (numpy.ndarray): Array to store actions for all episodes and stages.
        - all_rewards (numpy.ndarray): Array to store rewards for all episodes and stages.
        - all_next_states (numpy.ndarray): Array to store next states for all episodes and stages.
        - b (int): Current batch counter.
        """
        # ---------------------------------------
        # Environment Setup (Same from previous tasks)
        # ---------------------------------------
        self.model_name = "DP-LSVI-Luyo"
        self.H = H
        self.K = K
        self.d = 10
        self.num_actions = num_actions
        self.delta = delta
        self.eps = eps  # privacy parameter ε
        self.p = p   # failure probability p
        self.seed = seed
        self.action_seed = action_seed

        np.random.seed(self.seed)
        
        # Generate action vectors
        self.a_vectors = self.generate_action_vectors(num_actions=self.num_actions)
        
        # Initialize alpha sequence
        self.alpha_seq = [0] * self.H
        
        # ---------------------------------------
        # Parameters from Appendix E.1
        # ---------------------------------------
        # Given formulas:
        self.lambda_ = self.d
        # B = ceil((K eps)^{2/5} * d^{3/5} * H^{1/5})
        self.B = math.ceil(((self.K * self.eps) ** (2/5) * (self.d ** (3/5)) * (self.H ** (1/5))))
        
        self.B0 = math.ceil(math.log2(self.B) + 1)
        self.sigma_Lambda = (128 / self.eps) * math.sqrt(self.B * self.H * self.B0) * math.log2((32 * self.H * self.B0 * self.B) / self.delta)
        self.sigma_u = (128 / (self.eps * self.H)) * math.sqrt(self.H * self.B) * math.log2((32 * self.H * self.B0 * self.B) / self.delta)
        
        self.YJ = self.sigma_Lambda * self.B0 * (4 * math.sqrt(self.d) + 2 * math.log((6 * self.K * self.H) / self.p))
        self.cK = self.d * self.YJ
        self.CJ = self.sigma_u * (math.sqrt(self.d) + 2 * math.sqrt(math.log((6 * self.K * self.H * self.d) / self.p)))
        self.UK = max(1, 2 * self.H * math.sqrt(self.d * self.K) / (self.lambda_ + self.cK) + self.CJ / (self.lambda_ + self.cK))
        # Interpret "242·18" as 242 * 18
        self.chi = 242 * 18 * (self.K ** 2) * self.d * self.UK * self.H / self.p
        self.beta = 24 * self.H * math.sqrt(self.d * (self.lambda_ + self.cK)) * math.log(self.chi)
        
        # Projection function Π[0,H]
        # Defined as a method below
        
        # ---------------------------------------
        # Initialization
        # ---------------------------------------
        # Initialize ˜Λ0,h = λ I_d, ˜w0,h = 0_d for all h
        self.Lambda_tilde = [self.lambda_ * np.eye(self.d) for _ in range(self.H)]
        self.w_tilde = [np.zeros(self.d) for _ in range(self.H)]
        
        # Build H trees with 2B nodes each:
        # Each node is a dxd symmetric matrix ZΛ = (Z' + Z'^T)/2 with Z' ~ N(0,σΛ^2)
        self.H_trees = []
        for hh in range(self.H):
            nodes = [self.gauss_sym_matrix(self.d, self.sigma_Lambda) for _ in range(2 * self.B)]
            self.H_trees.append(nodes)
        
        # (η_i^h) arrays:
        # For i in [B], h in [H], η_i_h ∈ R^d with η_i_h ~ N(0,σ_u^2 I)
        self.eta = np.random.normal(0, self.sigma_u, size=(self.B, self.H, self.d))
        
        # Define k_i = i * ceil(K/B) + 1 for i in [0,B)
        self.ceil_K_B = math.ceil(self.K / self.B)
        self.k_indices = [i * self.ceil_K_B + 1 for i in range(self.B)]
        # Note: last batch might not be fully used if K/B not integer.
        
        # Data storage
        # We'll store all transitions in arrays:
        # After k episodes, we have lists s_{i,h}, a_{i,h}, r_{i,h}, s_{i,h+1}.
        self.all_states = np.zeros((self.K, self.H), dtype=int)
        self.all_actions = np.zeros((self.K, self.H), dtype=int)
        self.all_rewards = np.zeros((self.K, self.H))
        self.all_next_states = np.zeros((self.K, self.H), dtype=int)
        
        # We'll maintain batch counter b, start from b=0
        self.b = 0

    def get_model_name(self):
        """
        Retrieve the name of the model.

        Returns:
        - str: The model name.
        """
        return self.model_name

    def get_H(self):
        """
        Retrieve the horizon length.

        Returns:
        - int: The horizon length H.
        """
        return self.H

    def generate_action_vectors(self, num_actions=100):
        """
        Generate fixed action vectors for reproducibility.

        Parameters:
        - num_actions (int, optional): Number of actions. Defaults to 100.
        - seed (int, optional): Seed for random number generation. Defaults to 42.

        Returns:
        - numpy.ndarray: Array of action vectors with shape (num_actions, 8).
        """
        rng = np.random.RandomState(self.action_seed)
        return rng.choice([-1, 1], size=(num_actions, 8))

    @staticmethod
    def delta_indicator(s, a):
        """
        Indicator function that returns 1 if (s, a) == (0, 0), else 0.

        Parameters:
        - s (int): State.
        - a (int): Action.

        Returns:
        - int: Indicator value (1 or 0).
        """
        return 1 if (s == 0 and a == 0) else 0

    def phi_func(self, s, a):
        """
        Compute the feature vector phi(s, a).

        Parameters:
        - s (int): State.
        - a (int): Action.

        Returns:
        - numpy.ndarray: Feature vector of length d + 2 (self.d + 2).
        """
        dlt = self.delta_indicator(s, a)
        return np.concatenate([self.a_vectors[a], [dlt, 1 - dlt]])

    def mu_h(self, s, alpha_h):
        """
        Define the mu_h function for transition probabilities.

        Parameters:
        - s (int): State.
        - alpha_h (int): Parameter affecting transition probabilities.

        Returns:
        - numpy.ndarray: Mu vector of length d.
        """
        val1 = (1 - s) ** alpha_h
        val2 = s ** alpha_h
        vec = np.zeros(self.d)
        if self.d >= 9:
            vec[8] = val1
        if self.d >= 10:
            vec[9] = val2
        return vec

    def transition_prob(self, s, a, h, alpha_h=0):
        """
        Compute the transition probabilities for a given state-action pair at stage h.

        Parameters:
        - s (int): Current state.
        - a (int): Action taken.
        - h (int): Current stage (1-based index).
        - alpha_h (int, optional): Parameter affecting transition probabilities. Defaults to 0.

        Returns:
        - list of float: Probability distribution over next states [P(s'=0), P(s'=1)].
        """
        phi_sa = self.phi_func(s, a)
        mu0 = self.mu_h(0, alpha_h)
        mu1 = self.mu_h(1, alpha_h)
        p0 = np.dot(phi_sa, mu0)
        p1 = np.dot(phi_sa, mu1)
        if p0 + p1 == 0:
            return [0.5, 0.5]
        return [p0 / (p0 + p1), p1 / (p0 + p1)]

    def reward_func(self, s, a):
        """
        Compute the reward for a given state-action pair.

        Parameters:
        - s (int): State.
        - a (int): Action.

        Returns:
        - int: Reward (1 or 0).
        """
        return self.delta_indicator(s, a)

    def gauss_sym_matrix(self, d, scale):
        """
        Generate a symmetric Gaussian matrix.

        Parameters:
        - d (int): Dimension of the matrix.
        - scale (float): Standard deviation for the Gaussian distribution.

        Returns:
        - numpy.ndarray: Symmetric d x d matrix.
        """
        Z = np.random.normal(0, scale, size=(d, d))
        return (Z + Z.T) / 2

    def norm_phi_Lambda_inv(self, phi_vec, Lambda_mat):
        """
        Compute the norm of phi with respect to the inverse of Lambda.

        Parameters:
        - phi_vec (numpy.ndarray): Feature vector.
        - Lambda_mat (numpy.ndarray): Lambda matrix.

        Returns:
        - float: Norm value.
        """
        inv_L = np.linalg.inv(Lambda_mat)
        return np.sqrt(phi_vec @ inv_L @ phi_vec)

    def Q_value(self, phi_vec, w_vec, Lambda_mat):
        """
        Compute the Q-value for a given feature vector, weight vector, and Lambda matrix.

        Q(·,·) = clip(w^T phi + beta * ||phi||_Lambda^-1, [0,H])

        Parameters:
        - phi_vec (numpy.ndarray): Feature vector.
        - w_vec (numpy.ndarray): Weight vector.
        - Lambda_mat (numpy.ndarray): Lambda matrix.

        Returns:
        - float: Computed Q-value, clipped to be within [0, H].
        """
        val = w_vec.dot(phi_vec)
        bonus = self.beta * self.norm_phi_Lambda_inv(phi_vec, Lambda_mat)
        return self.proj_0_H(val + bonus)

    def get_action(self, s, w_list, Lambda_list):
        """
        Select the action with the highest Q-value for a given state.

        Parameters:
        - s (int): Current state.
        - w_list (list of numpy.ndarray): List of weight vectors for each stage.
        - Lambda_list (list of numpy.ndarray): List of Lambda matrices for each stage.

        Returns:
        - int: Selected action index.
        """
        best_a = 0
        best_val = -float('inf')
        for a_ in range(self.num_actions):
            phisa = self.phi_func(s, a_)
            val = self.Q_value(phisa, w_list, Lambda_list)
            if val > best_val:
                best_val = val
                best_a = a_
        return best_a

    def get_value(self, s, w_list, Lambda_list):
        """
        Compute the value of a state by taking the maximum Q-value over all actions.

        Parameters:
        - s (int): Current state.
        - w_list (list of numpy.ndarray): List of weight vectors for each stage.
        - Lambda_list (list of numpy.ndarray): List of Lambda matrices for each stage.

        Returns:
        - float: Maximum Q-value for the state.
        """
        best_val = -float('inf')
        for a_ in range(self.num_actions):
            phisa = self.phi_func(s, a_)
            val = self.Q_value(phisa, w_list, Lambda_list)
            if val > best_val:
                best_val = val
        return best_val

    def proj_0_H(self, x):
        """
        Clip the value x to be within [0, H].

        Parameters:
        - x (float): Value to clip.

        Returns:
        - float: Clipped value within [0, H].
        """
        return max(0, min(self.H, x))

    def get_epsilon(self):
        """
        Retrieve the epsilon (privacy budget) value.

        Returns:
        - float: The epsilon value.
        """
        return self.eps

    def run(self):
        """
        Execute the DP-LSVI-Luyo algorithm over K episodes.

        This method performs the main loop of the algorithm, iterating through each episode,
        generating trajectories, updating parameters in batches, and monitoring performance.
        """
        # Initialize variables for performance monitoring
        total_rewards = 0
        ep_rewards = []

        for k_ in range(1, self.K + 1):
            # User side:
            # Uses (˜Λ_b,h, ˜w_b,h) to select actions.
            s = np.random.choice([0, 1])
            for h_ in range(self.H):
                a = self.get_action(s, self.w_tilde[h_], self.Lambda_tilde[h_])
                r = self.reward_func(s, a)
                probs = self.transition_prob(s, a, h_ + 1, self.alpha_seq[h_])
                s_next = np.random.choice([0, 1], p=probs)
        
                self.all_states[k_ - 1, h_] = s
                self.all_actions[k_ - 1, h_] = a
                self.all_rewards[k_ - 1, h_] = r
                self.all_next_states[k_ - 1, h_] = s_next
        
                s = s_next
        
            # Update total rewards for performance monitoring
            episode_reward = sum(self.all_rewards[k_ - 1])
            total_rewards += episode_reward
            ep_rewards.append(episode_reward)
            average_reward = total_rewards / k_
        
            # Print performance monitoring information
            print(f"Episode {k_}/{self.K}, Average Reward So Far: {average_reward:.4f}")
        
            # Server side update:
            # Check if we reached the end of a batch:
            if (self.b < self.B) and (k_ + 1 == self.k_indices[self.b]):
                # Perform batch update
                # Compute from all data (1..k)
                # ˜w_{b+1,H+1}=0, Q_{k+1,H+1}=0 means terminal layer w=0
                # We'll do backward pass:
                # Data from episodes 1 to k:
                S = self.all_states[:k_, :]
                A = self.all_actions[:k_, :]
                R = self.all_rewards[:k_, :]
                S_next = self.all_next_states[:k_, :]
        
                # Determine prefix length for Hb+1_h:
                H_prefix_list = []
                for hh in range(self.H):
                    # sum the first (b+1) nodes:
                    prefix_mat = np.zeros((self.d, self.d))
                    for idx in range(self.b + 1):
                        prefix_mat += self.H_trees[hh][idx]
                    H_prefix_list.append(prefix_mat)
        
                # Compute Λ_{k+1,h} for all h
                Lambda_for_next = [None] * (self.H + 1)  # store Λ_{k+1,h} for reuse
                for hh in range(self.H):
                    phi_stack = []
                    for i_ep in range(k_):
                        ph = self.phi_func(S[i_ep, hh], A[i_ep, hh])
                        phi_stack.append(ph)
                    if len(phi_stack) > 0:
                        Phi_mat = np.array(phi_stack)  # shape (k_, d)
                        Lambda_kplus1_h = self.lambda_ * np.eye(self.d) + Phi_mat.T @ Phi_mat
                    else:
                        Lambda_kplus1_h = self.lambda_ * np.eye(self.d)
                    Lambda_for_next[hh] = Lambda_kplus1_h
                Lambda_for_next[self.H] = self.lambda_ * np.eye(self.d)  # terminal
        
                # Initialize new w
                w_new = [None] * (self.H + 1)
                w_new[self.H] = np.zeros(self.d)  # terminal
        
                # Backward pass to compute w_new
                for hh in range(self.H - 1, -1, -1):
                    # Compute V_{k+1,h+1}(s_{i,h+1}):
                    usum = np.zeros(self.d)
                    for i_ep in range(k_):
                        ph = self.phi_func(S[i_ep, hh], A[i_ep, hh])
                        s_next_h = S_next[i_ep, hh]
                        # Compute V_next
                        if hh == self.H - 1:
                            V_next = 0.0
                        else:
                            # Compute tilde_Lambda_{b+1,h+1}:
                            tilde_Lambda_next = Lambda_for_next[hh + 1] + (self.cK + self.YJ) * np.eye(self.d) + H_prefix_list[hh + 1]
        
                            # V_next = max_a Q from w_new[h+1], tilde_Lambda_next:
                            best_val = -float('inf')
                            for a_next in range(self.num_actions):
                                phisn = self.phi_func(s_next_h, a_next)
                                val = self.w_tilde[hh + 1].dot(phisn) + self.beta * self.norm_phi_Lambda_inv(phisn, tilde_Lambda_next)
                                val = self.proj_0_H(val)
                                if val > best_val:
                                    best_val = val
                            V_next = best_val
        
                        target = R[i_ep, hh] + V_next
                        usum += ph * target
        
                    if hh < self.H:
                        # Update Λ for current h
                        Phi_mat = np.array([self.phi_func(S[i_ep, hh], A[i_ep, hh]) for i_ep in range(k_)])
                        if Phi_mat.size > 0:
                            Lambda_h = Phi_mat.T @ Phi_mat + self.lambda_ * np.eye(self.d)
                        else:
                            Lambda_h = self.lambda_ * np.eye(self.d)
        
                        # Update w for current h
                        if Phi_mat.size > 0:
                            try:
                                w_h_new = np.linalg.inv(Lambda_h).dot(usum)
                            except np.linalg.LinAlgError:
                                # In case Lambda_h is singular, fallback to zeros
                                w_h_new = np.zeros(self.d)
                        else:
                            w_h_new = np.zeros(self.d)
        
                        w_new[hh] = w_h_new
                        Lambda_for_next[hh] = Lambda_h
                    else:
                        w_new[hh] = np.zeros(self.d)  # Terminal stage
        
                # Update global w_tilde and Lambda_tilde from w_new and tilde_Lambda
                for hh in range(self.H):
                    tilde_Lambda_h = Lambda_for_next[hh] + (self.cK + self.YJ) * np.eye(self.d) + H_prefix_list[hh]
                    self.w_tilde[hh] = w_new[hh]
                    self.Lambda_tilde[hh] = tilde_Lambda_h
        
                self.b += 1  # Move to next batch

    def compute_rewards(self):
        """
        Compute episodic rewards and the average reward across all episodes.

        Returns:
        - list of float: Total rewards for each episode.
        - float: Average reward across all episodes.
        """
        ep_rewards = [sum(self.all_rewards[i]) for i in range(self.K)]
        average_reward = np.mean(ep_rewards)
        return ep_rewards, average_reward

if __name__ == "__main__":
    dp_lsvi_luyo = DP_LSVI_Luyo()
    print(f"Model Name: {dp_lsvi_luyo.get_model_name()}")