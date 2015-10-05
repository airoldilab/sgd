#!/usr/bin/env Rscript
# SECTION 3.3, TABLE 4
# This is used to generate the table in M-estimation experiments section.

library(sgd)
library(ggplot2)

generate.data <- function(N, d) {
  l2 <- function(x) sqrt(sum(x**2))
  X <- matrix(rnorm(N*d, mean=0, sd=1/sqrt(N)), nrow=N, ncol=d)
  theta <- runif(d)
  theta <- theta * 6 *sqrt(d) / l2(theta)

  # noise
  ind <- rbinom(N, size=1, prob=.95)
  epsilon <- ind * rnorm(N) + (1-ind) * rep(10 ,N)

  Y <- X %*% theta + epsilon
  return(list(y=Y, X=X, theta=theta))
}

# Dimensions
N <- 1e4
d <- 5e2

# Generate data.
set.seed(42)
data <- generate.data(N, d)
dat <- data.frame(y=data$y, x=data$X)

times.sgd <- c()
times.aisgd <- c()
times.hqreg <- c()

nSim <- 10
for (i in 1:nSim) {
  sgd.theta1 <- sgd(y ~ .-1, data=dat, model="m",
    sgd.control=list(
    method="sgd",
    lr.control=c(15, NA, NA, 1/2), npass=1, pass=T, size=1, start=rep(5,d)))
  times.sgd <- c(times.sgd, sgd.theta1$times)
  sgd.theta2 <- sgd(y ~ .-1, data=dat, model="m",
    sgd.control=list(
    method="ai-sgd",
    lr.control=c(15, NA, NA, 2/3), npass=1, pass=T, size=1, start=rep(5,d)))
  times.aisgd <- c(times.aisgd, sgd.theta2$times)

  library(hqreg)
  time_start <- proc.time()[3]
  hqreg <- hqreg(data$X, as.vector(data$y), method = "huber",
    gamma=3, alpha=1)
  times.hqreg <- c(times.hqreg, as.numeric(proc.time()[3] - time_start))
}
print(mean(times.sgd))
print(mean(times.aisgd))
print(mean(times.hqreg))
