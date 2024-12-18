# wrapper_functions.R
library(reticulate)
library(here)
source(here::here("R/source_python_classes.R"))

# Wrapper function for DP_LSVI_Ngo
create_DP_LSVI_Ngo <- function(H = 10,
                               K = 100,
                               num_actions = 100,
                               p = 0.01,
                               rho = 10.0,
                               C = 2.0,
                               seed = 0,
                               action_seed = 42) {
  # Source the Python class
  source_python_class("DP_LSVI_Ngo.py", "DP_LSVI_Ngo")

  # Create an instance of the Python class with specified parameters
  dp_lsvi_ngo <- reticulate::import_main()$DP_LSVI_Ngo(
    H = as.integer(H),
    K = as.integer(K),
    num_actions = as.integer(num_actions),
    p = as.numeric(p),
    rho = as.numeric(rho),
    C = as.numeric(C),
    seed = as.integer(seed),
    action_seed = as.integer(action_seed)
  )

  # Define R wrapper functions for the class methods
  wrapper <- list(
    get_model_name = function() dp_lsvi_ngo$get_model_name(),
    get_H = function() dp_lsvi_ngo$get_H(),
    get_epsilon = function() dp_lsvi_ngo$get_epsilon(),
    run = function() dp_lsvi_ngo$run(),
    compute_rewards = function() {
      rewards <- dp_lsvi_ngo$compute_rewards()
      list(
        episodic_rewards = rewards[[1]],
        average_reward = rewards[[2]]
      )
    }
  )

  return(wrapper)
}

# Wrapper function for DP_LSVI_UCB_PlusPlus
create_DP_LSVI_UCB_PlusPlus <- function(H = 10,
                                        K = 100,
                                        num_actions = 100,
                                        delta = 0.01,
                                        rho = 10.0,
                                        seed = 0,
                                        action_seed = 42) {
  # Source the Python class
  source_python_class("DP_LSVI_UCB_PlusPlus.py", "DP_LSVI_UCB_PlusPlus")

  # Create an instance of the Python class with specified parameters
  dp_lsvi_ucb_plusplus <- reticulate::import_main()$DP_LSVI_UCB_PlusPlus(
    H = as.integer(H),
    K = as.integer(K),
    num_actions = as.integer(num_actions),
    delta = as.numeric(delta),
    rho = as.numeric(rho),
    seed = as.integer(seed),
    action_seed = as.integer(action_seed)
  )

  # Define R wrapper functions for the class methods
  wrapper <- list(
    get_model_name = function() dp_lsvi_ucb_plusplus$get_model_name(),
    get_H = function() dp_lsvi_ucb_plusplus$get_H(),
    get_epsilon = function() dp_lsvi_ucb_plusplus$get_epsilon(),
    run = function() dp_lsvi_ucb_plusplus$run(),
    compute_rewards = function() {
      rewards <- dp_lsvi_ucb_plusplus$compute_rewards()
      list(
        episodic_rewards = rewards[[1]],
        average_reward = rewards[[2]]
      )
    }
  )

  return(wrapper)
}

# Wrapper function for LSVI_UCB_PlusPlus
create_LSVI_UCB_PlusPlus <- function(H = 10,
                                     K = 100,
                                     num_actions = 100,
                                     delta = 0.01,
                                     seed = 0,
                                     action_seed = 42) {
  # Source the Python class
  source_python_class("LSVI_UCB_PlusPlus.py", "LSVI_UCB_PlusPlus")

  # Create an instance of the Python class with specified parameters
  lsvi_ucb_plusplus <- reticulate::import_main()$LSVI_UCB_PlusPlus(
    H = as.integer(H),
    K = as.integer(K),
    num_actions = as.integer(num_actions),
    delta = as.numeric(delta),
    seed = as.integer(seed),
    action_seed = as.integer(action_seed)
  )

  # Define R wrapper functions for the class methods
  wrapper <- list(
    get_model_name = function() lsvi_ucb_plusplus$get_model_name(),
    get_H = function() lsvi_ucb_plusplus$get_H(),
    run = function() lsvi_ucb_plusplus$run(),
    compute_rewards = function() {
      rewards <- lsvi_ucb_plusplus$compute_rewards()
      list(
        episodic_rewards = rewards[[1]],
        average_reward = rewards[[2]]
      )
    }
  )

  return(wrapper)
}

# Wrapper function for LSVI_UCB
create_LSVI_UCB <- function(H = 10,
                            K = 100,
                            num_actions = 100,
                            delta = 0.01,
                            lambda_reg = 1.0,
                            seed = 0,
                            action_seed = 42) {
  # Source the Python class
  source_python_class("LSVI_UCB.py", "LSVI_UCB")

  # Create an instance of the Python class with specified parameters
  lsvi_ucb <- reticulate::import_main()$LSVI_UCB(
    H = as.integer(H),
    K = as.integer(K),
    num_actions = as.integer(num_actions),
    delta = as.numeric(delta),
    lambda_reg = as.numeric(lambda_reg),
    seed = as.integer(seed),
    action_seed = as.integer(action_seed)
  )

  # Define R wrapper functions for the class methods
  wrapper <- list(
    get_model_name = function() lsvi_ucb$get_model_name(),
    get_H = function() lsvi_ucb$get_H(),
    run = function() lsvi_ucb$run(),
    compute_rewards = function() {
      rewards <- lsvi_ucb$compute_rewards()
      list(
        episodic_rewards = rewards[[1]],
        average_reward = rewards[[2]]
      )
    }
  )

  return(wrapper)
}

# Wrapper function for DP_LSVI_Luyo
create_DP_LSVI_Luyo <- function(H = 10,
                                K = 100,
                                num_actions = 100,
                                delta = 0.01,
                                eps = 10.0,
                                p = 0.05,
                                seed = 0,
                                action_seed = 42) {
  # Source the Python class
  source_python_class("DP_LSVI_Luyo.py", "DP_LSVI_Luyo")

  # Create an instance of the Python class with specified parameters
  dp_lsvi_luyo <- reticulate::import_main()$DP_LSVI_Luyo(
    H = as.integer(H),
    K = as.integer(K),
    num_actions = as.integer(num_actions),
    delta = as.numeric(delta),
    eps = as.numeric(eps),
    p = as.numeric(p),
    seed = as.integer(seed),
    action_seed = as.integer(action_seed)
  )

  # Define R wrapper functions for the class methods
  wrapper <- list(
    get_model_name = function() dp_lsvi_luyo$get_model_name(),
    get_H = function() dp_lsvi_luyo$get_H(),
    run = function() dp_lsvi_luyo$run(),
    compute_rewards = function() {
      rewards <- dp_lsvi_luyo$compute_rewards()
      list(
        episodic_rewards = rewards[[1]],
        average_reward = rewards[[2]]
      )
    }
  )

  return(wrapper)
}
