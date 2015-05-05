#!/usr/bin/env Rscript
# Demo usage of sgd for linear regression on simulated normal data.
#
# Data generating process:
#   Y = X %*% θ + ɛ, where
#     X ~ Normal(0, 1)
#     θ = (5,...,5)
#     ɛ ~ Normal(0,1)
# Dimensions:
#   N=1e4 observations
#   d=1e2 parameters

library(sgd)

# Dimensions
N <- 1e4
d <- 1e2

# Generate data.
X <- matrix(rnorm(N*d), ncol=d)
theta <- rep(5, d+1)
eps <- rnorm(N)
y <- cbind(1, X) %*% theta + eps
dat <- data.frame(y=y, x=X)

sgd.est <- sgd(y ~ ., data=dat, model="glm")
sgd.est$coefficients
