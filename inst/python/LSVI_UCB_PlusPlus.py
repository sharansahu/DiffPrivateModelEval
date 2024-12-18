import numpy as np
import math

class LSVI_UCB_PlusPlus:
    def __init__(self, 
                 H=10, 
                 K=100,  # Number of episodes
                 num_actions=100,
                 delta=0.01, 
                 seed=0, 
                 action_seed=42):
        """
        Initialize the LSVI-UCB++ Algorithm with default parameters.

        Parameters:
        - model_name (str): Name of the model.
        - H (int): Horizon length (number of stages per episode).
        - K (int): Number of episodes.
        - d (int): Dimension of the feature space.
        - num_actions (int): Number of possible actions.
        - delta (float): Confidence parameter.
        - lmbda (float): Regularization parameter.
        - a_vectors (numpy.ndarray): Generated action vectors.

        Initialization:
        - klast (int): Last episode index where a policy update occurred.
        - klast_Sigma (list of numpy.ndarray): List of Sigma matrices from the last update for each stage.
        - pQ (numpy.ndarray): Q-values for all episodes, stages, states, and actions.
                             Shape: (K+1, H+1, 2, num_actions)
        - qQ (numpy.ndarray): Alternative Q-values for all episodes, stages, states, and actions.
                             Shape: (K+1, H+1, 2, num_actions)
        - pV (numpy.ndarray): Value estimates for all episodes, stages, and states.
                             Shape: (K+1, H+1, 2)
        - qV (numpy.ndarray): Alternative value estimates for all episodes, stages, and states.
                             Shape: (K+1, H+1, 2)
        - S, A, R, S_next (numpy.ndarray): Arrays to store states, actions, rewards, and next states for all episodes and stages.
        - Sigma_list (list of numpy.ndarray): List of Sigma matrices for each stage.
        - beta, bar_beta, tilde_beta (float): Confidence bound parameters.
        - w_h_list (list of numpy.ndarray): List of weight vectors for each stage.
        - sigma_bar_history (numpy.ndarray or None): Array to store sigma_bar values, initialized as None.
        - C (float): Policy update rate threshold.
        - current_episode (int): Counter for the current episode.
        """
        # ---------------------------------------
        # Parameters
        # ---------------------------------------
        self.model_name = "LSVI-UCB++"
        self.H = H
        self.K = K
        self.d = 10
        self.num_actions = num_actions
        self.delta = delta
        self.seed = seed
        self.action_seed = action_seed
        
        np.random.seed(self.seed)
        
        # ---------------------------------------
        # Environment/Feature Map Setup
        # ---------------------------------------
        self.a_vectors = self.generate_action_vectors(num_actions=self.num_actions)
        self.lmbda = 1 / (self.H ** 2)
        
        # Initialize variables
        self.klast = 0
        self.klast_Sigma = [self.lmbda * np.eye(self.d) for _ in range(self.H)] 
        # For each h: Σ0,h = Σ1,h = λI (we interpret as Σ_{1,h} = λI)
        
        # Initialize Q and ˇQ
        # pQ: Q-values for all episodes, h, states, and actions
        # Shape: (K+1, H+1, 2, num_actions)
        self.pQ = np.ones((self.K+1, self.H+1, 2, self.num_actions)) * self.H
        self.qQ = np.zeros((self.K+1, self.H+1, 2, self.num_actions))
        
        # Initialize V and ˇV
        # Shape: (K+1, H+1, 2)
        self.pV = np.zeros((self.K+1, self.H+1, 2))
        self.qV = np.zeros((self.K+1, self.H+1, 2))
        self.pV[:, self.H, :] = 0
        self.qV[:, self.H, :] = 0
        
        # Initialize data storage
        # Shape: (K, H)
        self.S = np.zeros((self.K, self.H), dtype=int)
        self.A = np.zeros((self.K, self.H), dtype=int)
        self.R = np.zeros((self.K, self.H))
        self.S_next = np.zeros((self.K, self.H), dtype=int)
        
        # Initialize parameters for LSVI-UCB++
        # λ = 1/H^2
        
        # Confidence radii
        self.log_term = math.log((self.d * self.H * self.K) / (self.delta * self.lmbda))
        self.beta = self.H * math.sqrt(self.d * self.lmbda) + math.sqrt(self.d * (self.log_term ** 2))
        self.bar_beta = self.H * math.sqrt(self.d * self.lmbda) + math.sqrt((self.d ** 3) * (self.H ** 2) * (self.log_term ** 2))
        self.tilde_beta = (self.H ** 2) * math.sqrt(self.d * self.lmbda) + math.sqrt((self.d ** 3) * (self.H ** 4) * (self.log_term ** 2))
        
        # Initialize Sigma_list
        self.Sigma_list = [self.lmbda * np.eye(self.d) for _ in range(self.H)]
        
        # Initialize sigma_bar_history
        # Shape: (K, H)
        self.sigma_bar_history = None  # Will be initialized during the first episode
    
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
    
    def norm_phi_inv(self, phi, Sigma):
        """
        Compute the norm of phi with respect to the inverse of Sigma.

        ||phi||_{Sigma^{-1}} = sqrt(phi^T Sigma^{-1} phi)

        Parameters:
        - phi (numpy.ndarray): Feature vector.
        - Sigma (numpy.ndarray): Sigma matrix.

        Returns:
        - float: Norm value.
        """
        try:
            inv_Sigma = np.linalg.inv(Sigma)
        except np.linalg.LinAlgError:
            # Add a small value to the diagonal for numerical stability
            inv_Sigma = np.linalg.inv(Sigma + 1e-6 * np.eye(self.d))
        return math.sqrt(phi.dot(inv_Sigma).dot(phi))
    
    def compute_E(self, phi, Sigma):
        """
        Compute the E_k,h term used in the update equations.

        E_k,h = min{tilde_beta * ||phi||_{Sigma^{-1}}, H^2} + min{2 * H * bar_beta * ||phi||_{Sigma^{-1}}, H^2}

        Parameters:
        - phi (numpy.ndarray): Feature vector.
        - Sigma (numpy.ndarray): Sigma matrix.

        Returns:
        - float: Computed E_k,h value.
        """
        val = self.norm_phi_inv(phi, Sigma)
        part1 = min(self.tilde_beta * val, self.H ** 2)
        part2 = min(2 * self.H * self.bar_beta * val, self.H ** 2)
        return part1 + part2
    
    def compute_D(self, phi, w_hat, w_check, Sigma):
        """
        Compute the D_k,h term used in the update equations.

        D_k,h = min{4 * d^3 * H^2 * ((w_hat - w_check)^T phi + 2 * bar_beta * ||phi||_{Sigma^{-1}}, d^3 * H^3}

        Parameters:
        - phi (numpy.ndarray): Feature vector.
        - w_hat (numpy.ndarray): Weight vector for hat.
        - w_check (numpy.ndarray): Weight vector for check.
        - Sigma (numpy.ndarray): Sigma matrix.

        Returns:
        - float: Computed D_k,h value.
        """
        difference = w_hat - w_check
        val = difference.dot(phi) + 2 * self.bar_beta * self.norm_phi_inv(phi, Sigma)
        return min(4 * (self.d ** 3) * (self.H ** 2) * val, (self.d ** 3) * (self.H ** 3))
    
    def run(self):
        """
        Execute one episode of the LSVI-UCB++ algorithm.

        This method performs the backward pass to compute w_hat and w_check,
        checks if a policy update is needed based on the determinant condition,
        updates Q and qQ tables if necessary, and then executes the forward pass
        to simulate one episode and update Sigma matrices accordingly.

        Additionally, it prints the average reward achieved up to the current episode
        for performance monitoring.
        """
        total_rewards = 0
        ep_rewards = []
        for k in range(1, self.K + 1):
            # BACKWARD PASS:
            # Compute w_hat, w_check
            # w_hat_k,h = Σ^{-1}_{k,h} ∑_{i=1}^{k-1} ¯σ^{-2}_{i,h} φ(...) * V_k,h+1(...)
            # w_check_k,h = Σ^{-1}_{k,h} ∑_{i=1}^{k-1} ¯σ^{-2}_{i,h} φ(...) * ˇV_k,h+1(...)
            
            # Prepare to compute w_hat and w_check for each h
            w_hat = np.zeros((self.H, self.d))
            w_check = np.zeros((self.H, self.d))
            
            # Initialize sigma_bar_history if it's the first episode
            if self.sigma_bar_history is None:
                self.sigma_bar_history = np.ones((self.K, self.H)) * self.H  # Initialize to H
            
            # Iterate backwards through horizons
            for h_ in range(self.H - 1, -1, -1):
                Phi_mat = []
                p_target_vec = []
                q_target_vec = []
                
                for i_ep in range(k - 1):
                    ph = self.phi_func(self.S[i_ep, h_], self.A[i_ep, h_])
                    # V_k,h+1(s_{i,h+1}) and ˇV_k,h+1(s_{i,h+1})
                    if h_ == self.H - 1:
                        V_next = 0
                        V_check_next = 0
                    else:
                        s_next_ = self.S_next[i_ep, h_]
                        V_next = self.pV[k - 1, h_ + 1, s_next_]
                        V_check_next = self.qV[k - 1, h_ + 1, s_next_]
                    
                    # ¯σ_i,h from sigma_bar_history
                    sb = self.sigma_bar_history[i_ep, h_]
                    inv_sigma2 = 1 / (sb ** 2)
                    
                    Phi_mat.append(ph)
                    p_target_vec.append(V_next * inv_sigma2)
                    q_target_vec.append(V_check_next * inv_sigma2)
                
                Phi_mat = np.array(Phi_mat)
                p_target_vec = np.array(p_target_vec)
                q_target_vec = np.array(q_target_vec)
                
                if Phi_mat.size > 0:
                    # Compute w_hat and w_check
                    Sigma = self.Sigma_list[h_]
                    try:
                        inv_Sigma = np.linalg.inv(Sigma)
                    except np.linalg.LinAlgError:
                        inv_Sigma = np.linalg.inv(Sigma + 1e-6 * np.eye(self.d))  # Add small value for stability
                    
                    w_hat[h_] = inv_Sigma.dot(Phi_mat.T.dot(p_target_vec))
                    w_check[h_] = inv_Sigma.dot(Phi_mat.T.dot(q_target_vec))
                else:
                    w_hat[h_] = np.zeros(self.d)
                    w_check[h_] = np.zeros(self.d)
            
            # Check determinant condition:
            det_cur = [np.linalg.det(self.Sigma_list[h_]) for h_ in range(self.H)]
            if self.klast == 0:
                det_prev = [self.lmbda * np.linalg.det(np.eye(self.d)) for _ in range(self.H)]
            else:
                det_prev = [np.linalg.det(self.klast_Sigma[h_]) for h_ in range(self.H)]
            
            ratio_condition = any(det_cur[h_] >= 2 * det_prev[h_] for h_ in range(self.H))
            
            # If ratio_condition true, update Qk and ˇQk:
            if ratio_condition:
                for h_ in range(self.H):
                    for s_ in [0, 1]:
                        for a_ in range(self.num_actions):
                            ph = self.phi_func(s_, a_)
                            r_sa = self.reward_func(s_, a_)
                            E_kh = self.compute_E(ph, self.Sigma_list[h_])
                            D_kh = self.compute_D(ph, w_hat[h_], w_check[h_], self.Sigma_list[h_])
                            
                            if h_ == self.H - 1:
                                V_next = 0
                                V_check_next = 0
                            else:
                                s_next_ = self.S_next[k - 1, h_]
                                V_next = self.pV[k - 1, h_ + 1, s_next_]
                                V_check_next = self.qV[k - 1, h_ + 1, s_next_]
                            
                            # Update pQ and qQ
                            p_est = r_sa + w_hat[h_].dot(ph) + self.beta * math.sqrt(ph.dot(np.linalg.inv(self.Sigma_list[h_])).dot(ph))
                            pQ_val = min(p_est, self.pQ[k - 1, h_, s_, a_], self.H)
                            self.pQ[k, h_, s_, a_] = pQ_val
                            
                            q_est = r_sa + w_check[h_].dot(ph) - self.bar_beta * math.sqrt(ph.dot(np.linalg.inv(self.Sigma_list[h_])).dot(ph))
                            qQ_val = max(q_est, self.qQ[k - 1, h_, s_, a_], 0)
                            self.qQ[k, h_, s_, a_] = qQ_val
                            
                # Update klast and klast_Sigma
                self.klast = k
                self.klast_Sigma = [self.Sigma_list[h_].copy() for h_ in range(self.H)]
            else:
                # If no update, carry forward previous Q and ˇQ
                self.pQ[k, :, :, :] = self.pQ[k - 1, :, :, :]
                self.qQ[k, :, :, :] = self.qQ[k - 1, :, :, :]
            
            # Update V and ˇV
            for h_ in range(self.H):
                for s_ in [0, 1]:
                    self.pV[k, h_, s_] = np.max(self.pQ[k, h_, s_, :])
                    self.qV[k, h_, s_] = np.max(self.qQ[k, h_, s_, :])
            
            self.pV[k, self.H, :] = 0
            self.qV[k, self.H, :] = 0
            
            # FORWARD PASS:
            # Initialize state s randomly
            s = np.random.choice([0, 1])
            sigma_bar_current = np.ones(self.H) * self.H  # Placeholder for current episode
            
            for h_ in range(self.H):
                # a_k_h = argmax_a Qk,h(s_k,h,a)
                a = np.argmax(self.pQ[k, h_, s, :])
                r = self.reward_func(s, a)
                s_next = self.transition_func(s, a, h_ + 1)
                
                # Store transition data
                self.S[k - 1, h_] = s
                self.A[k - 1, h_] = a
                self.R[k - 1, h_] = r
                self.S_next[k - 1, h_] = s_next
                
                # Compute E_k,h and D_k,h
                ph = self.phi_func(s, a)
                E_kh = self.compute_E(ph, self.Sigma_list[h_])
                D_kh = self.compute_D(ph, w_hat[h_], w_check[h_], self.Sigma_list[h_])
                
                # Compute σ_k,h
                if h_ == self.H - 1:
                    V_next = 0
                    V_check_next = 0
                else:
                    V_next = self.pV[k, h_ + 1, s_next]
                    V_check_next = self.qV[k, h_ + 1, s_next]
                
                sigma_kh = math.sqrt(V_next * V_check_next + E_kh + D_kh + self.H)
                
                # Update Sigma_list
                inv_Sigma = np.linalg.inv(self.Sigma_list[h_])
                norm_14 = self.norm_phi_inv(ph, self.Sigma_list[h_])
                sigma_bar = max(sigma_kh, self.H, 2 * (self.d ** 3) * (self.H ** 2) * norm_14)
                sigma_bar_current[h_] = sigma_bar
                
                # Update Σ_{k+1,h}:
                inv_sigma2 = 1 / (sigma_bar ** 2)
                self.Sigma_list[h_] = self.Sigma_list[h_] + inv_sigma2 * np.outer(ph, ph)
                
                # Transition to next state
                s = s_next
            
            # Store sigma_bar for this episode:
            if k == 1:
                self.sigma_bar_history = sigma_bar_current.reshape((1, self.H))
            else:
                self.sigma_bar_history = np.vstack([self.sigma_bar_history, sigma_bar_current])
            
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
    lsvi_ucb_plusplus = LSVI_UCB_PlusPlus()
    print(f"Model Name: {lsvi_ucb_plusplus.get_model_name()}")