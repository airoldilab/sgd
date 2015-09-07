#!/usr/bin/env Rscript
# Demo usage of sgd for M-estimation using the Huber loss.
# Example taken from Donoho and Montanari (2013, Section 2.4).
#
# Data generating process:
#   Y = X %*% theta + epsilon, where
#     X ~ Normal(0, 1/N)
#     theta ~ Unif([0,1]^d) with fixed 2-norm 6*sqrt(d)
#     epsilon ~ ContaminatedNormal(0.05, 10) = 0.95z + 0.05h_{10},
#       where z ~ Normal(0,1) and h_x = unit atom at x
#
# Dimensions:
#   N=1000 observations
#   d=200 parameters

library(sgd)
library(ggplot2)

generate.data <- function(N, d) {
  library(mvtnorm)
  l2 <- function(x) sqrt(sum(x**2))
  X <- rmvnorm(N, mean=rep(0, d), sigma=diag(d)/N)
  theta <- runif(d)
  theta <- theta * 6 *sqrt(d) / l2(theta)

  # noise
  ind <- rbinom(N, size=1, prob=.95)
  epsilon <- ind * rnorm(N) + (1-ind) * rep(10 ,N)

  Y <- X %*% theta + epsilon
  return(list(y=Y, X=X, theta=theta))
}

# Dimensions
N <- 1000
d <- 200

# Generate data.
set.seed(42)
data <- generate.data(N, d)
dat <- data.frame(y=data$y, x=data$X)

sgd.theta <- sgd(y ~ .-1, data=dat, model="m", sgd.control=list(method="sgd",
  lr.control=c(15, NA, NA, 1/2), npass=10, pass=T))

plot(sgd.theta, data$theta, label="sgd", type="mse-param") +
  geom_hline(yintercept=1.5, color="green")
