#!/usr/bin/env Rscript
# Demo usage of sgd for fitting generalized linear model with Poisson
# link function with simulated data.
#
# Data generating process:
#   X is either (0, 0), (1, 0), (0, 1) with 0.6, 0.2, 0.2 probability
#   theta = (log(2), log(4))
#   y = Poisson(exp(X %*% theta))
#
# Dimensions:
#   N=1e5 observations
#   d=2 parameters

library(sgd)

# Dimensions
N <- 1e5

# Generate data.
set.seed(42)
Q <- 0.2
code <- sample(0:2, size=N, replace=T, prob=c((1-2*Q), Q, Q))
X <- matrix(0, nrow=N, ncol=2)
X[,1] <- as.numeric(code==1)
X[,2] <- as.numeric(code==2)
theta <- matrix(c(log(2), log(4)), ncol=1)
y <- matrix(rpois(N, exp(X %*% theta)), ncol=1)
dat <- data.frame(y=y, x=X)

sgd.theta <- sgd(y ~ .-1, data=dat, model="glm",
                 model.control=list(family=poisson()))
mean((sgd.theta$coefficients - theta)^2) # MSE
