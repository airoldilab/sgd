#!/usr/bin/env Rscript
# Demo usage of sgd for estimating parameters of a normal distribution using
# the method of moments.
#
# Data generating process:
#   X ~ Normal(4, 2)
#
# Dimensions:
#   N=1e3 observations
#   d=2   parameters

library(sgd)

# Dimensions
N <- 1e3
d <- 2
theta <- c(4, 2)

# Generate data.
set.seed(42)
X <- matrix(rnorm(N, mean=theta[1], sd=theta[2]), ncol=1)

# Gradient of moment function (using 3 moments)
gr <- function(theta, x) {
  return(as.matrix(c(
    mean(2*(theta[1] - x) +
         2*(theta[2]^2 - (x - theta[1])^2) * 2*(-theta[1] +x) +
         2*(x^3 - theta[1]*(theta[1]^2 + 3*theta[2]^2)) * (-3*theta[1]^2 -
           3*theta[2]^2)),
    mean(0 +
         2*(theta[2]^2 - (x - theta[1])^2) * 2*theta[2] +
         2*(x^3 - theta[1]*(theta[1]^2 + 3*theta[2]^2)) * -6*theta[1]*theta[2])
    )))
}
sgd.theta <- sgd(X, y=matrix(NA, nrow=nrow(X)), model="gmm",
  model.control=list(gr=gr, nparams=2),
  sgd.control=list(method="sgd", npasses=100, lr="adagrad"))
sgd.theta
