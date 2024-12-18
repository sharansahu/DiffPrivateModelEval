devtools::install_github("sharansahu/DiffPrivateModelEval")
library(DiffPrivateModelEval)
library(ggplot2)

dp_algo <- create_DP_LSVI_Ngo(H = 15, K = 50)
lsvi_algo <- create_LSVI_UCB(H = 15, K = 50)
dp_algo1 <- create_DP_LSVI_UCB_PlusPlus(H = 15, K = 50)
lsvi_algo1 <- create_LSVI_UCB_PlusPlus(H = 15, K = 50)

# Run the algorithms
dp_algo$run()
lsvi_algo$run()
dp_algo1$run()
lsvi_algo1$run()

# Plot multiple algorithm comparison
plot_multiple_algos(list(dp_algo, lsvi_algo, dp_algo1, lsvi_algo1))
