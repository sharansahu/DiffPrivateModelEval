# DiffPrivateModelEval: An R Package for Differentially Private Model Evaluation
This repository contains an R package designed to facilitate the evaluation of reinforcement learning algorithms with differential privacy guarantees. 
The package provides seamless integration with Python-based algorithms using the reticulate package, robust visualization tools for cumulative regret analysis, 
and easy-to-use wrapper functions for algorithms like DP-LSVI-Ngo, DP-LSVI-UCB++, and more.

## Key Features:
- Privacy-Aware RL Algorithms: Includes implementations of state-of-the-art differentially private reinforcement learning methods.
- Cumulative Regret Analysis: Intuitive plotting functions to compare and analyze algorithm performance.
- Scalable and Modular: Easily extendable for additional algorithms or custom evaluation scenarios.

## Installation

To install the package from GitHub, use the following commands:

```r
# Install devtools if not already installed
install.packages("devtools")

# Install DiffPrivateModelEval from GitHub
devtools::install_github("sharansahu/DiffPrivateModelEval")

# Load the package
library(DiffPrivateModelEval)
```

## Example Usage
Here’s an example demonstrating the use of the package to run and compare the algorithms:

```r
# Load the required libraries
library(DiffPrivateModelEval)
library(ggplot2)

# Create algorithm instances
dp_algo <- create_DP_LSVI_Ngo(H = 15, K = 50)
lsvi_algo <- create_LSVI_UCB(H = 15, K = 50)
dp_algo1 <- create_DP_LSVI_UCB_PlusPlus(H = 15, K = 50)
lsvi_algo1 <- create_LSVI_UCB_PlusPlus(H = 15, K = 50)

# Run the algorithms
dp_algo$run()
lsvi_algo$run()
dp_algo1$run()
lsvi_algo1$run()
```

## Visualizing Performance
To visualize the cumulative regret for a single algorithm:

```r
# Plot cumulative regret for DP-LSVI-Ngo
plot_regret(dp_algo)
```

To visualize the cumulative regret for multiple algorithms:

```r
plot_multiple_algos(list(dp_algo, lsvi_algo, dp_algo1, lsvi_algo1))
```
