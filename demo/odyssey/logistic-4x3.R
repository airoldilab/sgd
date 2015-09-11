# Run experiments on four data sets and export the results.

library(ggplot2)
library(gridExtra)
library(sgd)

source("demo/odyssey/logistic-covtype.R")
source("demo/odyssey/logistic-delta.R")
source("demo/odyssey/logistic-sido.R")
source("demo/odyssey/logistic-mnist.R")

pdf("img/exp_4x3.pdf")
grid.arrange(out_covtype[[1]], out_covtype[[2]], out_covtype[[3]],
             out_delta[[1]], out_delta[[2]], out_delta[[3]],
             out_sido[[1]], out_sido[[2]], out_sido[[3]],
             out_mnist[[1]], out_mnist[[2]], out_mnist[[3]],
             ncol=3)
dev.off()
