import numpy as np

class LSVI_UCB:
    def __init__(self, 
                 H=10, 
                 K=100,  # Number of episodes 
                 num_actions=100,
                 delta=0.01, 
                 lambda_reg = 1.0, 
                 seed=0, 
                 action_seed=42):
        """
        Initialize the LSVI-UCB Algorithm with default parameters.
        
        Environment Setup:
        - model_name (str): Name of the model.
        - H (int): Horizon length (number of stages per episode).
        - K (int): Number of episodes.
        - d (int): Dimension of the feature space.
        - num_actions (int): Number of possible actions.
        - delta (float): Confidence parameter.
        - lambda_reg (float): Regularization parameter for ridge regression.
        - a_vectors (numpy.ndarray): Generated action vectors.
        - alpha_seq (list): Sequence of alpha parameters for transition probabilities.
        - beta (float): Confidence bound parameter.
        - data (list): Storage for all transitions across episodes.
        - w_h (list of numpy.ndarray): Weight vectors for each stage.
        - Lambda_h_list (list of numpy.ndarray): Regularization matrices for each stage.
        """
        # Environment setup:
        self.model_name = "LSVI-UCB"
        self.H = H
        self.K = K
        self.d = 10
        self.num_actions = num_actions
        self.delta = delta
        self.lambda_reg = lambda_reg
        self.seed = seed
        self.action_seed = action_seed

        np.random.seed(self.seed)
        
        # Generate action vectors
        self.a_vectors = self.generate_action_vectors(num_actions=self.num_actions)
        
        # Initialize alpha sequence
        self.alpha_seq = [0] * self.H
        
        # Define beta:
        self.T = self.K * self.H
        self.iota = np.log((2 * self.d * self.T) / self.delta)
        self.c = 1.0
        self.beta = self.c * self.d * self.H * np.sqrt(self.iota)
        
        # Data structure: to store all transitions for episodes
        self.data = []
        
        # Initialize parameters
        self.w_h = [np.zeros(self.d) for _ in range(self.H + 1)]  # w_{h}, h=1..H
        self.Lambda_h_list = [self.lambda_reg * np.eye(self.d) for _ in range(self.H + 1)]
    
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
        - seed (int, optional): Seed for random number generation. Defaults to 0.
        
        Returns:
        - numpy.ndarray: Array of action vectors with shape (num_actions, 8).
        """
        rng = np.random.RandomState(self.action_seed)
        return rng.choice([-1,1], size=(num_actions,8))
    
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
    
    def phi(self, s, a):
        """
        Compute the feature vector phi(s, a).
        
        Parameters:
        - s (int): State.
        - a (int): Action.
        
        Returns:
        - numpy.ndarray: Feature vector of length 10 (self.d).
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
        val1 = (1 - s) ^ alpha_h
        val2 = s ^ alpha_h
        vec = np.zeros(self.d)
        vec[8] = val1
        vec[9] = val2
        return vec
    
    def transition_prob(self, s, a, h, alpha_h):
        """
        Compute the transition probabilities for a given state-action pair at stage h.
        
        Parameters:
        - s (int): Current state.
        - a (int): Action taken.
        - h (int): Current stage.
        - alpha_h (int): Parameter affecting transition probabilities.
        
        Returns:
        - list of float: Probability distribution over next states [P(s'=0), P(s'=1)].
        """
        phi_sa = self.phi(s, a)
        mu0 = self.mu_h(0, alpha_h)
        mu1 = self.mu_h(1, alpha_h)
        p0 = np.dot(phi_sa, mu0)
        p1 = np.dot(phi_sa, mu1)
        if p0 + p1 == 0:
            return [0.5, 0.5]
        return [p0 / (p0 + p1), p1 / (p0 + p1)]
    
    def reward(self, s, a):
        """
        Compute the reward for a given state-action pair.
        
        Parameters:
        - s (int): State.
        - a (int): Action.
        
        Returns:
        - int: Reward (1 or 0).
        """
        return self.delta_indicator(s, a)
    
    def compute_Q(self, h, s, a, w_h, Lambda_h):
        """
        Compute the Q-value for a given stage, state, and action.
        
        Parameters:
        - h (int): Current stage.
        - s (int): Current state.
        - a (int): Action.
        - w_h (numpy.ndarray): Weight vector for stage h.
        - Lambda_h (numpy.ndarray): Regularization matrix for stage h.
        
        Returns:
        - float: Computed Q-value, clipped to be at most H.
        """
        phisa = self.phi(s, a)
        inv_Lambda = np.linalg.inv(Lambda_h)
        mean_val = w_h.dot(phisa)
        bonus = self.beta * np.sqrt(phisa.dot(inv_Lambda).dot(phisa))
        return min(mean_val + bonus, self.H)
    
    def run_episode(self, w_list, Lambda_list):
        """
        Run a single episode using the current policy derived from Q-values.
        
        Parameters:
        - w_list (list of numpy.ndarray): Current weight vectors for all stages.
        - Lambda_list (list of numpy.ndarray): Current regularization matrices for all stages.
        
        Returns:
        - list of tuples: Trajectory of the episode, where each tuple contains (state, action, reward, next_state).
        """
        # Start from s_1 uniform in {0,1}
        s = np.random.choice([0,1])
        traj = []
        for h in range(1, self.H + 1):
            # Choose a = argmax_a Q_h(s,a)
            Lambda_h = Lambda_list[h]
            best_val = None
            best_a = 0
            for a_ in range(self.num_actions):
                val = self.compute_Q(h, s, a_, w_list[h], Lambda_h)
                if (best_val is None) or (val > best_val):
                    best_val = val
                    best_a = a_
            a = best_a
            r = self.reward(s, a)
            probs = self.transition_prob(s, a, h, self.alpha_seq[h-1])
            s_next = np.random.choice([0,1], p=probs)
            traj.append((s, a, r, s_next))
            s = s_next
        return traj
    
    def run(self):
        """
        Execute the LSVI-UCB algorithm over K episodes.
        
        This method performs the main loop of the algorithm, iterating through each episode,
        generating trajectories, updating Q and V tables, and monitoring performance.
        """
        # Initialize variables for performance monitoring
        total_rewards = 0
        ep_rewards = []
        
        for k in range(1, self.K + 1):
            # Run episode k using current w_h and Lambda_h_list
            traj = self.run_episode(self.w_h, self.Lambda_h_list)
            self.data.append(traj)
        
            # Update total rewards for performance monitoring
            episode_reward = sum([x[2] for x in traj])
            total_rewards += episode_reward
            ep_rewards.append(episode_reward)
            average_reward = total_rewards / k
            
            # Print performance monitoring information
            print(f"Episode {k}/{self.K}, Average Reward So Far: {average_reward:.4f}")
        
            # After collecting new data, update w_h and Lambda_h for next iteration
            # Algorithm line: For h=H..1:
            #   Λ_h = sum φφ^T + λI
            #   w_h = Λ_h^{-1} sum φ [r + max_a Q_{h+1}]
        
            # Initialize new parameters
            new_w = [self.w_h[i].copy() for i in range(self.H + 1)]
            new_Lambda = [self.Lambda_h_list[i].copy() for i in range(self.H + 1)]
        
            for h in range(self.H, 0, -1):
                Phi_list = []
                Y_list = []
                all_phis = []
                for ep in range(k):
                    (s_h, a_h, r_h, s_next_h) = self.data[ep][h-1]
                    phisa = self.phi(s_h, a_h)
                    all_phis.append(phisa)
                    if h == self.H:
                        val_next = 0
                    else:
                        Lambda_hplus1 = new_Lambda[h+1]
                        w_hplus1 = new_w[h+1]
                        inv_Lambda_hplus1 = np.linalg.inv(Lambda_hplus1)
                        best_val = None
                        for a_next in range(self.num_actions):
                            phisn = self.phi(s_next_h, a_next)
                            mean_val = w_hplus1.dot(phisn)
                            bonus = self.beta * np.sqrt(phisn.dot(inv_Lambda_hplus1).dot(phisn))
                            q_val = min(mean_val + bonus, self.H)
                            if best_val is None or q_val > best_val:
                                best_val = q_val
                        val_next = best_val
                    target = r_h + val_next
                    Phi_list.append(phisa)
                    Y_list.append(target)
        
                if len(Phi_list) > 0:
                    Phi_mat = np.array(Phi_list)
                    Y_vec = np.array(Y_list)
                    Lambda_h = Phi_mat.T.dot(Phi_mat) + self.lambda_reg * np.eye(self.d)
                    try:
                        w_h_new = np.linalg.inv(Lambda_h).dot(Phi_mat.T.dot(Y_vec))
                    except np.linalg.LinAlgError:
                        # In case Lambda_h is singular, fallback to zeros
                        w_h_new = np.zeros(self.d)
                else:
                    Lambda_h = self.lambda_reg * np.eye(self.d)
                    w_h_new = np.zeros(self.d)
        
                new_w[h] = w_h_new
                new_Lambda[h] = Lambda_h
        
            self.w_h = new_w
            self.Lambda_h_list = new_Lambda
    
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
    lsvi_ucb = LSVI_UCB()
    print(f"Model Name: {lsvi_ucb.get_model_name()}")