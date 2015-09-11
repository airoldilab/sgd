#!/usr/bin/env Rscript
# Demo usage of sgd for linear regression on simulated normal data.
#
# Data generating process:
#   Y = Binomial(N, p), where
#     p = 1 / (1 + exp(-X %*% theta + epsilon):
#     X ~ Normal(0, 1)
#     theta = (5,...,5)
#     epsilon ~ Normal(0,1)
#
# Dimensions:
#   N=1e5 observations
#   d=5 parameters

library(sgd)

# Dimensions
N <- 1e5
d <- 5

# Generate data.
set.seed(42)
X <- matrix(rnorm(N*d), ncol=d)
theta <- rep(5, d+1)
eps <- rnorm(N)
p <- 1/(1+exp(-(cbind(1, X) %*% theta + eps)))
y <- rbinom(N, 1, p)
dat <- data.frame(y=y, x=X)

sgd.theta <- sgd(y ~ ., data=dat, model="glm",
                 model.control=list(family="binomial"),
                 sgd.control=list(lr.control=c(100, NA, NA, NA), npasses=1,
                 pass=T))

plot(sgd.theta, theta, label="ai-sgd", type="mse-param")
