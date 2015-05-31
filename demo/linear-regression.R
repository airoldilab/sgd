#!/usr/bin/env Rscript
# Demo usage of sgd for linear regression on simulated normal data.
#
# Data generating process:
#   Y = X %*% theta + epsilon, where
#     X ~ Normal(0, 1)
#     theta = (5,...,5)
#     epsilon ~ Normal(0,1)
#
# Dimensions:
#   N=1e5 observations
#   d=1e2 parameters

library(sgd)

# Dimensions
N <- 1e5
d <- 1e2

# Generate data.
set.seed(42)
X <- matrix(rnorm(N*d), ncol=d)
theta <- rep(5, d+1)
eps <- rnorm(N)
y <- cbind(1, X) %*% theta + eps
dat <- data.frame(y=y, x=X)

sgd.theta <- sgd(y ~ ., data=dat, model="lm")
mean((sgd.theta$coefficients - theta)^2) # MSE
