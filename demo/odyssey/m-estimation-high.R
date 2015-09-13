#!/usr/bin/env Rscript

library(sgd)
library(ggplot2)

generate.data <- function(N, d, theta) {
  X <- matrix(rnorm(N*d, mean=0, sd=1/sqrt(N)), nrow=N, ncol=d)

  # noise
  ind <- rbinom(N, size=1, prob=.95)
  epsilon <- ind * rnorm(N) + (1-ind) * rep(10 ,N)

  Y <- X %*% theta + epsilon
  return(list(y=Y, X=X))
}

# Dimensions
N <- 100
d <- 200
nstreams <- 10 # number of streams

set.seed(42)
# Generate truth.
l2 <- function(x) sqrt(sum(x**2))
theta <- runif(d)
theta <- theta * 6 *sqrt(d) / l2(theta)

for (nstream in 1:nstreams) {
  # Generate stream of data.
  data <- generate.data(N, d, theta)
  dat <- data.frame(y=data$y, x=data$X)

  if (nstream == 1) {
    start <- rep(5, d)
  } else {
    start <- aisgd.theta$coefficients
  }
  aisgd.theta <- sgd(y ~ .-1, data=dat, model="m",
    sgd.control=list(
    method="ai-sgd",
    lr.control=c(15, NA, NA, 2/3), npass=1, pass=T, size=0.5*N, start=start,
    start.idx=(nstream-1)*N+1))
  if (nstream == 1) {
    times.aisgd <- aisgd.theta$times
    pos.aisgd <- aisgd.theta$pos
    estimates.aisgd <- aisgd.theta$estimates
  } else {
    times.aisgd <- c(times.aisgd, aisgd.theta$times + max(times.aisgd))
    pos.aisgd <- c(pos.aisgd, aisgd.theta$pos + max(pos.aisgd))
    estimates.aisgd <- cbind(estimates.aisgd, aisgd.theta$estimates)
  }

  if (nstream == 1) {
    start <- rep(5, d)
  } else {
    start <- sgd.theta$coefficients
  }
  sgd.theta <- sgd(y ~ .-1, data=dat, model="m",
    sgd.control=list(
    method="sgd",
    lr.control=c(15, NA, NA, 1/2), npass=1, pass=T, size=0.5*N, start=start,
    start.idx=(nstream-1)*N+1))
  if (nstream == 1) {
    times.sgd <- sgd.theta$times
    pos.sgd <- sgd.theta$pos
    estimates.sgd <- sgd.theta$estimates
  } else {
    times.sgd <- c(times.sgd, sgd.theta$times + max(times.sgd))
    pos.sgd <- c(pos.sgd, sgd.theta$pos + max(pos.sgd))
    estimates.sgd <- cbind(estimates.sgd, sgd.theta$estimates)
  }
}

# Create sgd object as if the command were run once.
obj.aisgd <- list()
obj.aisgd$times <- times.aisgd
obj.aisgd$pos <- pos.aisgd
obj.aisgd$estimates <- estimates.aisgd
class(obj.aisgd) <- "sgd"

obj.sgd <- list()
obj.sgd$times <- times.sgd
obj.sgd$pos <- pos.sgd
obj.sgd$estimates <- estimates.sgd
class(obj.sgd) <- "sgd"

objs.sgd <- list("ai-sgd"=obj.aisgd, sgd=obj.sgd)

save(objs.sgd, theta, file="out/m-estimation-high.RData")
#plot(objs.sgd, theta, type="mse-param")
