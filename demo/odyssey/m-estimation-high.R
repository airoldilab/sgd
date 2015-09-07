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
#   N=1e6 observations
#   d=1e4 parameters

library(sgd)

generate.data <- function(N, d) {
  l2 <- function(x) sqrt(sum(x**2))
  X <- matrix(rnorm(N*d, mean=0, sd=1/N), nrow=N, ncol=d)
  theta <- runif(d)
  theta <- theta * 6 *sqrt(d) / l2(theta)

  # noise
  ind <- rbinom(N, size=1, prob=.95)
  epsilon <- ind * rnorm(N) + (1-ind) * rep(10 ,N)

  Y <- X %*% theta + epsilon
  return(list(y=Y, X=X, theta=theta))
}

# Dimensions
N <- 1e6
d <- 1e4

# Generate data.
set.seed(42)
data <- generate.data(N, d)
dat <- data.frame(y=data$y, x=data$X)

job.id <- as.integer(commandArgs(trailingOnly = TRUE))
if (job.id == 1) {
  sgd.theta <- sgd(y ~ .-1, data=dat, model="m", sgd.control=list(
    method="sgd", lr="adagrad",
    lr.control=c(5, NA), npass=10, pass=T, size=N, start=rep(5, d)))
} else if (job.id == 2) {
  sgd.theta <- sgd(y ~ .-1, data=dat, model="m", sgd.control=list(
    method="ai-sgd", lr="adagrad",
    lr.control=c(5, NA), npass=10, pass=T, size=N, start=rep(5, d)))
}

# Save outputs into individual files.
save(sgd.theta, file=sprintf("out/m-estimation-high-%i.RData", job.id))
