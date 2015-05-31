# Run experiments on four data sets and export the results.

library(ggplot2)
library(gridExtra)
library(sgd)

source("script/exp_covtype.R")
source("script/exp_delta.R")
source("script/exp_sido.R")
source("script/exp_mnist.R")

pdf("img/exp_4x3.pdf")
grid.arrange(out_covtype[[1]], out_covtype[[2]], out_covtype[[3]],
             out_delta[[1]], out_delta[[2]], out_delta[[3]],
             out_sido[[1]], out_sido[[2]], out_sido[[3]],
             out_mnist[[1]], out_mnist[[2]], out_mnist[[3]],
             ncol=3)
dev.off()
