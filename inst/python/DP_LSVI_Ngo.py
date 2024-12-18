from math import ceil, log, sqrt
import numpy as np
import math

class DP_LSVI_Ngo:
    def __init__(self, 
                 H=10, 
                 K=100,  # Number of episodes
                 num_actions=100,
                 p=0.01,
                 rho=10.0,
                 C=2.0,
                 seed=0, 
                 action_seed=42):
        """
        Initialize the DP-LSVI-Ngo Algorithm with default parameters.

        Environment and Feature Map Setup:
        - model_name (str): Name of the model.
        - H (int): Horizon length (number of stages per episode).
        - K (int): Number of episodes.
        - d (int): Dimension of the feature space.
        - num_actions (int): Number of possible actions.
        - p (float): Failure probability.
        - rho (float): Privacy parameter.
        - C (float): Policy update rate.
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
        - Lambda_list (list of numpy.ndarray): List of regularization matrices for each stage.
        - C_mats (list of numpy.ndarray): List of cumulative matrices for each stage.
        - ktilde (int): Current update index.
        - Ncount (int): Update counter.
        - w_h_list (list of numpy.ndarray): List of weight vectors for each update.
        - Q_h_list (list of list of callable): List of Q-value functions for each update.
        - S, A, R, S_next (numpy.ndarray): Arrays to store states, actions, rewards, and next states for all episodes and stages.
        - det_init, det_prev (list of float): Lists to store initial and previous determinants of Lambda matrices.
        """
        # ---------------------------------------
        # Environment and Feature Map Setup
        # ---------------------------------------
        self.model_name = "DP-LSVI-Ngo"
        self.H = H
        self.K = K
        self.d = 10
        self.num_actions = num_actions
        self.p = p
        self.rho = rho
        self.C=C
        self.seed = seed
        self.action_seed = action_seed

        np.random.seed(self.seed)
        
        # Generate action vectors
        self.a_vectors = self.generate_action_vectors()
        
        # Define functions
        # These are kept as methods for better structure
        
        # Initialize Parameters
        self.lamb = 1.0

        self.eps = ceil(self.rho + 2*sqrt(self.rho*log(1 / self.p)))
        
        # Compute tilde_lambda_Lambda
        self.log_K = math.log(self.K)
        term_log = math.log((3 * self.K * self.H) / self.p)
        self.tilde_lambda_Lambda = (self.log_K * ((6 * math.sqrt(self.d) + 1) + 2 * term_log)) / math.sqrt(2 * self.rho)
        
        # Compute inner = 1 + (K / (tilde_lambda_Lambda * d))
        inner = 1 + (self.K / (self.tilde_lambda_Lambda * self.d))
        
        # Compute Nmax = int(d * H * log2(log(inner)))
        self.Nmax = int(self.d * self.H * math.log2(math.log(inner)))
        
        # Compute tilde_lambda_y
        log_inner = math.log(inner)
        log2_log = math.log2(log_inner)
        self.tilde_lambda_y = math.sqrt((self.H * self.d * self.H * (log2_log ** 2) * math.log((3 * self.K * self.H) / self.p)) / self.rho)
        
        # Compute UK
        self.UK = max(1, 2 * self.H * math.sqrt(self.d * self.K) / self.tilde_lambda_Lambda + self.tilde_lambda_y / self.tilde_lambda_Lambda)
        
        # Compute chi
        self.chi = (254 * 162) * (self.K ** 4) * self.d * self.UK * self.H / self.p
        
        # Compute beta
        log_chi = math.log(self.chi)
        self.beta = 5 * (self.H ** 2) * math.sqrt(self.d * self.tilde_lambda_Lambda) * log_chi + 6 * self.d * self.H * math.sqrt(log_chi)
        
        # Initialize Lambda_list
        self.Lambda_list = [2 * self.tilde_lambda_Lambda * np.eye(self.d) for _ in range(self.H)]
        
        # Initialize C_mats
        self.C_mats = [np.zeros((self.d, self.d)) for _ in range(self.H)]
        
        # Initialize counters
        self.ktilde = 1
        self.Ncount = 1
        
        # Initialize w_h_list and Q_h_list
        self.w_h_list = [np.zeros((self.H, self.d))]  # List of w_h arrays for each update
        self.Q_h_list = [[lambda x, a: self.H for _ in range(self.H)]]  # Dummy initialization
        
        # Initialize Data Storage
        self.S = np.zeros((self.K, self.H), dtype=int)
        self.A = np.zeros((self.K, self.H), dtype=int)
        self.R = np.zeros((self.K, self.H))
        self.S_next = np.zeros((self.K, self.H), dtype=int)
        
        # Initialize determinants
        self.det_init = [np.linalg.det(self.Lambda_list[h]) for h in range(self.H)]
        self.det_prev = self.det_init.copy()
    
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
    
    def get_epsilon(self):
        """
        Retrieve the epsilon (privacy budget) value.

        Returns:
        - float: The epsilon value.
        """
        return self.eps
    
    def generate_action_vectors(self):
        """
        Generate fixed action vectors for reproducibility.

        Parameters:
        - num_actions (int, optional): Number of actions. Defaults to 100.
        - seed (int, optional): Seed for random number generation. Defaults to 42.

        Returns:
        - numpy.ndarray: Array of action vectors with shape (num_actions, 8).
        """
        rng = np.random.RandomState(self.action_seed)
        return rng.choice([-1, 1], size=(self.num_actions, 8))
    
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
    
    def transition_func(self, s, a, h):
        """
        Compute the next state based on the current state, action, and stage.

        This function currently implements a simple random transition. 
        It can be modified to incorporate more complex transition dynamics.

        Parameters:
        - s (int): Current state.
        - a (int): Action taken.
        - h (int): Current stage (1-based index).

        Returns:
        - int: Next state (0 or 1).
        """
        # Simple random transition as defined
        return np.random.choice([0, 1])
    
    def proj_0_H(self, x):
        """
        Clip the value x to be within [0, H].

        Parameters:
        - x (float): Value to clip.

        Returns:
        - float: Clipped value within [0, H].
        """
        return max(0, min(self.H, x))
    
    def Q_func_factory(self, w_h_current, Lambda_list_current):
        """
        Factory method to create a Q-value function based on current weights and Lambda matrices.

        Q(h, s, a) = clip(w^T phi + beta * ||phi||_{Lambda^{-1}}, [0, H])

        Parameters:
        - w_h_current (numpy.ndarray): Current weight vector for all stages (shape: H x d).
        - Lambda_list_current (list of numpy.ndarray): Current Lambda matrices for all stages.

        Returns:
        - callable: A function that computes Q-value given h, s, a.
        """
        def Q(h, s, a):
            """
            Compute the Q-value for a given stage, state, and action.

            Parameters:
            - h (int): Current stage (0-based index).
            - s (int): Current state.
            - a (int): Action.

            Returns:
            - float: Computed Q-value, clipped to be within [0, H].
            """
            phi = self.phi_func(s, a)
            w = w_h_current[h]
            val = np.dot(w, phi)
            try:
                inv_Lambda = np.linalg.inv(Lambda_list_current[h])
            except np.linalg.LinAlgError:
                # If Lambda is singular, add a small identity for numerical stability
                inv_Lambda = np.linalg.inv(Lambda_list_current[h] + 1e-6 * np.eye(self.d))
            val_inside = np.dot(phi, np.dot(inv_Lambda, phi))
            # Clamp to avoid negative sqrt:
            val_inside = max(val_inside, 0.0)
            bonus = self.beta * math.sqrt(val_inside)
            return self.proj_0_H(val + bonus)
        
        return Q
    
    def run(self):
        """
        Execute the DP-LSVI-Ngo algorithm over K episodes.

        This method performs the main loop of the algorithm, iterating through each episode,
        generating trajectories, updating parameters in batches, and monitoring performance.
        """
        total_rewards = 0
        ep_rewards = []

        # Initialize Q_current
        w_current = self.w_h_list[0].copy()
        Q_current = self.Q_func_factory(w_current, self.Lambda_list)
        
        for k in range(1, self.K + 1):
            # Check determinant condition
            if k > 1:
                det_current = [np.linalg.det(self.Lambda_list[h]) for h in range(self.H)]
                ratio_condition = any((det_current[h] / self.det_prev[h]) >= self.C for h in range(self.H))
            else:
                ratio_condition = False
            
            if k > 1 and ratio_condition and self.Ncount < self.Nmax:
                # Policy update
                w_new = np.zeros((self.H, self.d))
                
                # Compute y_k_h for each h
                for h_ in range(self.H - 1, -1, -1):
                    Y_vec = np.zeros(self.d)
                    for i_ep in range(k - 1):
                        ph = self.phi_func(self.S[i_ep, h_], self.A[i_ep, h_])
                        # V^k_{h+1}(x_{i,h+1}) = max_a Q_current(h+1, x_{i,h+1}, a) if h+1 < H else 0
                        if h_ == self.H - 1:
                            V_next = 0.0
                        else:
                            s_next_ = self.S_next[i_ep, h_]
                            best_val = 0.0
                            for a_next in range(self.num_actions):
                                q_val = Q_current(h_ + 1, s_next_, a_next)
                                if q_val > best_val:
                                    best_val = q_val
                            V_next = best_val
                        Y_vec += ph * (self.R[i_ep, h_] + V_next)
                    
                    # Add noise
                    noise_scale_y = math.sqrt(6 * (self.H ** 2) * self.Nmax / self.rho)
                    y_tilde = Y_vec + np.random.normal(0, noise_scale_y, size=self.d)
                    
                    # Compute w_k_h
                    try:
                        inv_Lambda = np.linalg.inv(self.Lambda_list[h_])
                    except np.linalg.LinAlgError:
                        inv_Lambda = np.linalg.inv(self.Lambda_list[h_] + 1e-6 * np.eye(self.d))
                    w_k_h = np.dot(inv_Lambda, y_tilde)
                    w_new[h_] = w_k_h
                
                # Update ktilde and Ncount
                self.ktilde = k
                self.Ncount += 1
                
                # Update Q_current with new w
                Q_current = self.Q_func_factory(w_new, self.Lambda_list)
                
                # Store updated w_h
                self.w_h_list.append(w_new.copy())
                
                # Update determinants for next ratio check
                self.det_prev = [np.linalg.det(self.Lambda_list[h]) for h in range(self.H)]
            
            # Forward pass: Execute one episode
            # Initialize state s randomly
            s = np.random.choice([0, 1])
            for h_ in range(self.H):
                # Select action a = argmax_a Q_current(h_, s, a)
                best_val = -1e9
                best_a = 0
                for a_ in range(self.num_actions):
                    val = Q_current(h_, s, a_)
                    if val > best_val:
                        best_val = val
                        best_a = a_
                
                a = best_a
                r = self.reward_func(s, a)
                s_next = self.transition_func(s, a, h_ + 1)
                
                # Store transition data
                self.S[k - 1, h_] = s
                self.A[k - 1, h_] = a
                self.R[k - 1, h_] = r
                self.S_next[k - 1, h_] = s_next
                
                # Update C_mats
                ph = self.phi_func(s, a)
                outer_prod = np.outer(ph, ph)
                
                # Add Gaussian noise
                cov_noise = np.random.normal(0, math.sqrt((self.log_K ** 2) / (2 * self.rho)), size=(self.d, self.d))
                cov_noise = (cov_noise + cov_noise.T) / 2  # Symmetrize
                self.C_mats[h_] += outer_prod + cov_noise
                
                # Update Lambda_list
                self.Lambda_list[h_] = self.C_mats[h_] + 2 * self.tilde_lambda_Lambda * np.eye(self.d)
                
                # Transition to next state
                s = s_next

            episode_reward = np.sum(self.R[k - 1])
            total_rewards += episode_reward
            ep_rewards.append(episode_reward)
            average_reward = total_rewards / k
            
            # Print performance monitoring information
            print(f"Episode {k}/{self.K}, Average Reward So Far: {average_reward:.4f}")


    def compute_rewards(self):
        """
        Compute episodic rewards and the average reward across all episodes.

        Returns:
        - list of float: Total rewards for each episode.
        - float: Average reward across all episodes.
        """
        ep_rewards = [np.sum(self.R[i]) for i in range(self.K)]
        average_reward = np.mean(ep_rewards)
        return ep_rewards, average_reward

if __name__ == "__main__":
    dp_lsvi_ngo = DP_LSVI_Ngo()
    print(f"Model Name: {dp_lsvi_ngo.get_model_name()}")