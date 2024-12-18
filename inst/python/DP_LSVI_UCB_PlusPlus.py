from math import ceil, log, sqrt
import numpy as np

class DP_LSVI_UCB_PlusPlus:
    def __init__(self, 
                 H=10, 
                 K=100,  # Number of episodes
                 num_actions=100, 
                 delta=0.01, 
                 rho=10.0, 
                 seed=0, 
                 action_seed=42):
        """
        Initialize the DP-LSVI-UCB++ Algorithm with given parameters.
        
        Parameters:
        - H (int): Horizon length.
        - K (int): Number of episodes.
        - d (int): Dimension of feature space.
        - num_actions (int): Number of possible actions.
        - delta (float): Confidence parameter.
        - rho (float): Total zCDP budget.
        - seed (int): Seed for random number generation.
        - action_seed (int): Seed for action vector generation.
        """
        # Parameters
        self.model_name = "DP-LSVI-UCB++"
        self.H = H
        self.K = K
        self.d = 10
        self.num_actions = num_actions
        self.delta = delta
        self.rho = rho
        self.seed = seed
        self.action_seed = action_seed
        
        # Set random seed for reproducibility
        np.random.seed(self.seed)
        
        # Compute L and tilde_lambda_Lambda from given formulas
        self.compute_constants()
        
        # Initialize rho0, phi1, phi2, phi3, Z, K1
        self.initialize_random_variables()
        
        # Initialize tilde_Lambda, tilde_w_hat, tilde_w_check, tilde_w_bar for each stage h
        self.initialize_tilde_variables()
        
        # Initialize Q and V tables
        self.initialize_QV_tables()
        
        # Generate action vectors
        self.a_vectors = self.generate_action_vectors()
        
        # Data storage for episodes
        self.data = []
        
        # Initialize k_last
        self.k_last = 0
    
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
    
    def compute_constants(self):
        """
        Compute L, tilde_lambda_Lambda, beta_hat, beta_bar, and beta based on provided formulas.
        
        This method calculates various constants required for the algorithm based on the
        parameters H, K, d, rho, and delta.
        """
        H, K, d, rho, delta = self.H, self.K, self.d, self.rho, self.delta

        self.eps = ceil(self.rho + 2*sqrt(self.rho*log(1 / self.delta)))
        
        # Compute L
        log_term1 = np.log((10 * d * K * H) / delta)
        self.L = 4 * H * np.sqrt((d * H * K / rho) * log_term1)
        
        # Compute tilde_lambda_Lambda
        c1, c2 = 1.0, 1.0  # Constants for tilde_lambda_Lambda from O() terms
        log_term2 = np.log((5 * c1 * H) / delta)
        self.tilde_lambda_Lambda = np.sqrt((8 * d * H * K) / rho) * (2 + ((log_term2 / (c2 * d)) ** (2/3)))
        
        # Compute beta_hat
        log_term3 = np.log((H * (K**4) * (self.L**2) * d) / (delta * self.tilde_lambda_Lambda))
        self.beta_hat = H * self.L * np.sqrt(d * self.tilde_lambda_Lambda) + np.sqrt((d**3) * (self.H**2) * (log_term3**2))
        self.beta_check = self.beta_hat  # As per original script
        
        # Compute beta_bar
        self.beta_bar = (self.H**2) * (self.L**2) * np.sqrt(d * self.tilde_lambda_Lambda) + \
                        np.sqrt((d**3) * (self.H**4) * (log_term3**2))
        
        # Compute beta
        log_term4 = np.log(1 + (self.H * (self.K**4) * (self.L**2) * d) / (self.delta * self.tilde_lambda_Lambda))
        self.beta = self.H * self.L * np.sqrt(d * self.tilde_lambda_Lambda) + np.sqrt(d * (log_term4**2))
    
    def get_epsilon(self):
        """
        Retrieve the epsilon value computed during initialization.
        
        Returns:
        - int: The epsilon value.
        """
        return self.eps
    
    def initialize_random_variables(self):
        """
        Initialize random variables rho0, phi1, phi2, phi3, Z, and K1.
        
        This method sets up the initial random variables required for the algorithm,
        ensuring reproducibility by using the specified random seeds.
        """
        H, K, d, rho = self.H, self.K, self.d, self.rho
        rho0 = rho / (4 * H * K)
        self.rho0 = rho0
        
        self.phi1 = np.random.normal(0, np.sqrt(2 * (H**2) / rho0), size=d)
        self.phi2 = np.random.normal(0, np.sqrt(2 * (H**2) / rho0), size=d)
        self.phi3 = np.random.normal(0, np.sqrt(2 * (H**4) / rho0), size=d)
        
        Z = np.random.normal(0, np.sqrt(1 / (4 * rho0)), size=(d, d))
        self.K1 = (Z + Z.T) / np.sqrt(2)
    
    def initialize_tilde_variables(self):
        """
        Initialize tilde_Lambda, tilde_w_hat, tilde_w_check, and tilde_w_bar for each stage h.
        
        This method sets up the initial tilde variables for all stages in the horizon.
        """
        H, tilde_lambda_Lambda, d = self.H, self.tilde_lambda_Lambda, self.d
        
        self.tilde_Lambda = [2 * tilde_lambda_Lambda * np.eye(d) for _ in range(H)]
        self.tilde_w_hat = [np.zeros(d) for _ in range(H)]
        self.tilde_w_check = [np.zeros(d) for _ in range(H)]
        self.tilde_w_bar = [np.zeros(d) for _ in range(H)]
    
    def initialize_QV_tables(self):
        """
        Initialize Q and V tables.
        
        This method sets up the Q-value tables (pQ and qQ) and the value tables (pV and qV)
        for all episodes, stages, and actions. Initial Q-values are set to H, and V-values are set to zero.
        """
        H, K, d, num_actions = self.H, self.K, self.d, self.num_actions
        
        # Initialize Q tables: pQ and qQ
        self.pQ = np.ones((K + 1, H + 1, 2, num_actions)) * self.H
        self.qQ = np.zeros((K + 1, H + 1, 2, num_actions))
        
        # Initialize V tables: pV and qV
        self.pV = np.zeros((K + 1, H + 1, 2))
        self.qV = np.zeros((K + 1, H + 1, 2))
        self.pV[:, H, :] = 0
        self.qV[:, H, :] = 0
    
    def generate_action_vectors(self, num_actions=None, seed=None):
        """
        Generate fixed action vectors for reproducibility.
        
        Parameters:
        - num_actions (int, optional): Number of actions. Defaults to self.num_actions.
        - seed (int, optional): Seed for random number generation. Defaults to self.action_seed.
        
        Returns:
        - numpy.ndarray: Array of action vectors with shape (num_actions, 8).
        """
        if num_actions is None:
            num_actions = self.num_actions
        if seed is None:
            seed = self.action_seed
        rng = np.random.RandomState(seed)
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
        - numpy.ndarray: Feature vector of length 10 (self.d).
        """
        return np.concatenate([self.a_vectors[a], 
                               [self.delta_indicator(s, a), 
                                1 - self.delta_indicator(s, a)]])
    
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
    
    def norm_phi_Lambda_inv(self, phi, Lambda):
        """
        Compute the norm of phi with respect to the inverse of Lambda.
        
        Parameters:
        - phi (numpy.ndarray): Feature vector.
        - Lambda (numpy.ndarray): Lambda matrix.
        
        Returns:
        - float: Norm value.
        """
        invL = np.linalg.inv(Lambda)
        return np.sqrt(phi.dot(invL).dot(phi))
    
    @staticmethod
    def clip_0_H(x, H):
        """
        Clip the value x to be within [0, H].
        
        Parameters:
        - x (float): Value to clip.
        - H (int): Upper bound.
        
        Returns:
        - float: Clipped value within [0, H].
        """
        return max(0, min(H, x))
    
    def get_action_from_pQ(self, k, h, s):
        """
        Select the action with the highest Q-value from pQ.
        
        Parameters:
        - k (int): Current episode index.
        - h (int): Current stage.
        - s (int): Current state.
        
        Returns:
        - int: Selected action index.
        """
        vals = self.pQ[k, h, s, :]
        return np.argmax(vals)
    
    def mu_h(self, sprime, alpha=0):
        """
        Define the mu_h function for transition probabilities.
        
        Parameters:
        - sprime (int): Next state.
        - alpha (int, optional): Parameter (unused, set to 0). Defaults to 0.
        
        Returns:
        - numpy.ndarray: Mu vector of length d.
        """
        val1 = (1 - sprime) ** alpha
        val2 = sprime ** alpha
        vv = np.zeros(self.d)
        vv[8] = val1
        vv[9] = val2
        return vv
    
    def run(self):
        """
        Execute the regret minimization algorithm over K episodes.
        
        This method performs the main loop of the algorithm, iterating through each episode,
        generating trajectories, updating Q and V tables, and monitoring performance.
        """
        H, K, d, rho = self.H, self.K, self.d, self.rho
        L, tilde_lambda_Lambda = self.L, self.tilde_lambda_Lambda
        beta_hat, beta_check, beta_bar, beta = self.beta_hat, self.beta_check, self.beta_bar, self.beta
        phi1, phi2, phi3 = self.phi1, self.phi2, self.phi3
        K1 = self.K1
        tilde_Lambda = self.tilde_Lambda
        tilde_w_hat = self.tilde_w_hat
        tilde_w_check = self.tilde_w_check
        tilde_w_bar = self.tilde_w_bar
        a_vectors = self.a_vectors
        data = self.data
        pQ, qQ = self.pQ, self.qQ
        pV, qV = self.pV, self.qV
        
        # Initialize variables for performance monitoring
        total_rewards = 0
        ep_rewards = []
        
        for k_ in range(1, K + 1):
            # Forward pass: receive s_k^1
            s = np.random.choice([0, 1])
            traj = []
            
            # Generate trajectory for episode k_
            for h_ in range(H):
                a = self.get_action_from_pQ(k_ - 1, h_, s)  # use previous iteration Q
                r = self.reward_func(s, a)
                
                # Construct phi(s,a)
                phi_sa = self.phi_func(s, a)
                
                # Compute transition probabilities from linear MDP
                mu0 = self.mu_h(0, 0)
                mu1 = self.mu_h(1, 0)
                p0 = phi_sa.dot(mu0)
                p1 = phi_sa.dot(mu1)
                
                if p0 + p1 == 0:
                    probs = [0.5, 0.5]
                else:
                    probs = [p0 / (p0 + p1), p1 / (p0 + p1)]
                
                s_next = np.random.choice([0, 1], p=probs)
                traj.append((s, a, r, s_next))
                s = s_next
            
            data.append(traj)
            
            # Update total rewards for performance monitoring
            episode_reward = sum([x[2] for x in traj])
            total_rewards += episode_reward
            ep_rewards.append(episode_reward)
            average_reward = total_rewards / k_
            
            # Print performance monitoring information
            print(f"Episode {k_}/{K}, Average Reward So Far: {average_reward:.4f}")
            
            # Backward pass (only if k_ > 1)
            if k_ > 1:
                # Retrieve previous V-values
                pV_prev = pV[k_ - 1]
                qV_prev = qV[k_ - 1]
                
                # Initialize lists for each stage h
                Phi_all = [[] for _ in range(H)]
                p_val_all = [[] for _ in range(H)]
                q_val_all = [[] for _ in range(H)]
                w_val_all = [[] for _ in range(H)]
                
                # Assume sigma_bar = H for weighting
                inv_sigma2 = 1.0 / (H**2)
                
                # Aggregate data from all previous episodes
                for ep_i in range(k_ - 1):
                    ep_data = data[ep_i]
                    for h_ in range(H):
                        s_i, a_i, r_i, s_next_i = ep_data[h_]
                        phisa = self.phi_func(s_i, a_i)
                        
                        if h_ == H - 1:
                            p_t = 0
                            q_t = 0
                        else:
                            p_t = pV_prev[h_ + 1, s_next_i]
                            q_t = qV_prev[h_ + 1, s_next_i]
                        w_t = p_t ** 2
                        
                        Phi_all[h_].append(phisa)
                        p_val_all[h_].append(p_t)
                        q_val_all[h_].append(q_t)
                        w_val_all[h_].append(w_t)
                
                # Solve for w_hat, w_check, w_bar for each stage h
                for h_ in range(H):
                    Phi_mat = np.array(Phi_all[h_]) if len(Phi_all[h_]) > 0 else np.zeros((0, d))
                    if Phi_mat.shape[0] > 0:
                        Lambda_kh = 2 * tilde_lambda_Lambda * np.eye(d) + K1.copy()
                        for ph in Phi_mat:
                            Lambda_kh += inv_sigma2 * np.outer(ph, ph)
                        
                        # Correctly compute the confidence-weighted norm
                        norm_ph = np.linalg.norm(Phi_mat, axis=1)
                        b_p = inv_sigma2 * Phi_mat.T.dot(np.array(p_val_all[h_])) + self.phi1
                        b_q = inv_sigma2 * Phi_mat.T.dot(np.array(q_val_all[h_])) + self.phi2
                        b_w = inv_sigma2 * Phi_mat.T.dot(np.array(w_val_all[h_])) + self.phi3
                        
                        # Solve linear systems
                        try:
                            w_hat_h = np.linalg.solve(Lambda_kh, b_p)
                            w_check_h = np.linalg.solve(Lambda_kh, b_q)
                            w_bar_h = np.linalg.solve(Lambda_kh, b_w)
                        except np.linalg.LinAlgError:
                            # In case Lambda_kh is singular, fallback to zeros
                            w_hat_h = np.zeros(d)
                            w_check_h = np.zeros(d)
                            w_bar_h = np.zeros(d)
                        
                        tilde_w_hat[h_] = w_hat_h
                        tilde_w_check[h_] = w_check_h
                        tilde_w_bar[h_] = w_bar_h
                    else:
                        # If no data, keep weights as zero
                        tilde_w_hat[h_] = np.zeros(d)
                        tilde_w_check[h_] = np.zeros(d)
                        tilde_w_bar[h_] = np.zeros(d)
                
                # Check determinant condition
                # As per original script, always trigger Q update after first iteration
                if k_ == 2:
                    det_condition = True
                else:
                    det_condition = True
                
                if det_condition:
                    for h_ in range(H):
                        for s_ in [0, 1]:
                            for a_ in range(self.num_actions):
                                ph = self.phi_func(s_, a_)
                                
                                # Compute the norm with respect to tilde_Lambda
                                norm_phi_inv = self.norm_phi_Lambda_inv(ph, tilde_Lambda[h_])
                                
                                r_sa = self.reward_func(s_, a_)
                                
                                # Correctly compute confidence bounds using the inverse norm
                                p_conf = self.beta_hat * norm_phi_inv
                                q_conf = self.beta_hat * norm_phi_inv
                                
                                p_est = r_sa + tilde_w_hat[h_].dot(ph) + p_conf
                                q_est = r_sa + tilde_w_check[h_].dot(ph) + q_conf
                                
                                # Update pQ and qQ with proper clipping
                                pQ[k_, h_, s_, a_] = self.clip_0_H(min(p_est, pQ[k_ - 1, h_, s_, a_], self.H), self.H)
                                qQ[k_, h_, s_, a_] = min(q_est, qQ[k_ - 1, h_, s_, a_], 0)
                    
                    # Update k_last
                    self.k_last = k_
                else:
                    # If determinant condition not met, retain previous Q-values
                    self.pQ[k_, :, :, :] = self.pQ[k_ - 1, :, :, :]
                    self.qQ[k_, :, :, :] = self.qQ[k_ - 1, :, :, :]
                
                # Update V-values based on updated Q-values
                for h_ in range(H):
                    for s_ in [0, 1]:
                        self.pV[k_, h_, s_] = np.max(self.pQ[k_, h_, s_, :])
                        self.qV[k_, h_, s_] = np.max(self.qQ[k_, h_, s_, :])
                
                # Ensure terminal values are zero
                self.pV[k_, H, :] = 0
                self.qV[k_, H, :] = 0
    
    def compute_rewards(self):
        """
        Compute episodic rewards and the average reward across all episodes.
        
        Returns:
        - list of int: Total rewards for each episode.
        - float: Average reward across all episodes.
        """
        ep_rewards = [sum([x[2] for x in ep]) for ep in self.data]
        average_reward = np.mean(ep_rewards)
        return ep_rewards, average_reward

if __name__ == "__main__":
    dp_lsvi_ucb_plusplus = DP_LSVI_UCB_PlusPlus()
    print(f"Model Name: {dp_lsvi_ucb_plusplus.get_model_name()}")