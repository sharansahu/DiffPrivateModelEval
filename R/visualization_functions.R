library(reticulate)
library(ggplot2)

# Single Algorithm Cumulative Regret Plot
plot_regret <- function(algo) {
  # Retrieve rewards and compute cumulative regret
  rewards <- algo$compute_rewards()
  ep_rewards <- as.numeric(rewards$episodic_rewards)
  H <- algo$get_H()
  optimal_ep_reward <- H / 2.0
  ep_regret <- optimal_ep_reward - ep_rewards
  cumulative_regret <- cumsum(ep_regret)
  cumulative_regret <- abs(cumulative_regret)  # Ensure non-negative cumulative regret

  # Determine the title, including epsilon if available
  title <- paste("Cumulative Regret of", algo$get_model_name())
  if (inherits(algo, "python.builtin.object") && !is.null(algo$get_epsilon())) {
    eps <- algo$get_epsilon()
    title <- paste(title, "(ε =", eps, ")")
  }

  # Create the plot
  plot_data <- data.frame(
    Episode = seq_along(cumulative_regret),
    CumulativeRegret = cumulative_regret
  )

  ggplot(plot_data, aes(x = Episode, y = CumulativeRegret)) +
    geom_line(color = "blue", linewidth = 1) +
    labs(
      title = title,
      x = "Number of Episodes (K)",
      y = "Cumulative Regret"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(size = 14, face = "bold"),
      axis.title = element_text(size = 12)
    )
}

# Multiple Algorithm Comparison Plot
plot_multiple_algos <- function(algo_instances) {
  plot_data <- data.frame()

  # Validate that all algorithms have the same H and epsilon
  eps <- NULL
  H <- NULL

  for (algo in algo_instances) {
    if (inherits(algo, "python.builtin.object") && !is.null(algo$get_epsilon())) {
      eps_algo <- algo$get_epsilon()
      H_algo <- algo$get_H()
      if (is.null(eps)) {
        eps <- eps_algo
      } else if (eps != eps_algo) {
        stop("All algorithms must have the same privacy budget (ε) for comparison!")
      }
      if (is.null(H)) {
        H <- H_algo
      } else if (H != H_algo) {
        stop("All algorithms must have the same horizon length (H) for comparison!")
      }
    } else {
      H_algo <- algo$get_H()
      if (is.null(H)) {
        H <- H_algo
      } else if (H != H_algo) {
        stop("All algorithms must have the same horizon length (H) for comparison!")
      }
    }

    # Compute cumulative regret for the algorithm
    rewards <- algo$compute_rewards()
    ep_rewards <- as.numeric(rewards$episodic_rewards)
    optimal_ep_reward <- H / 2.0
    ep_regret <- optimal_ep_reward - ep_rewards
    cumulative_regret <- cumsum(ep_regret)
    cumulative_regret <- abs(cumulative_regret)

    algo_data <- data.frame(
      Episode = seq_along(cumulative_regret),
      CumulativeRegret = cumulative_regret,
      Algorithm = algo$get_model_name()
    )
    plot_data <- rbind(plot_data, algo_data)
  }

  # Determine the title, including epsilon if available
  title <- "Comparison of Cumulative Regret Bounds"
  if (!is.null(eps)) {
    title <- paste(title, "(ε =", eps, ")")
  }

  # Create the plot
  ggplot(plot_data, aes(x = Episode, y = CumulativeRegret, color = Algorithm)) +
    geom_line(linewidth = 1.2) +
    labs(
      title = title,
      x = "Number of Episodes (K)",
      y = "Cumulative Regret"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(size = 14, face = "bold"),
      axis.title = element_text(size = 12),
      legend.position = "top"
    ) +
    scale_color_brewer(palette = "Set1")
}
