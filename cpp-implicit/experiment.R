# Copyright (c) 2013
# Panos Toulis, ptoulis@fas.harvard.edu
#
library(mvtnorm)  # recall rmvnorm(n,...) returns n x p matrix.

base.learning.rate <- function(t, gamma0, alpha, c) {
  # Computes a learning rate of the form g * (1 + a * t * g)^-c
  #
  # Typically a, g have to be set according to the curvature 
  # of the loss function (log-likelihood)
  # c = is usually problem-independent but may get different values 
  # according to convexity of loss.
  #CHECK_TRUE(all(c(gamma0, alpha, c) >= 0), msg="Positive params in learning rate.")
  #CHECK_INTERVAL(c, 0, 1, msg="c in [0,1]")
  x = exp(log(gamma0) - c * log(1 + alpha * gamma0 * t))
  y = gamma0 * (1 + alpha * gamma0 * t)^-c
  #CHECK_NEAR(x, y, tol=1e-4)
  return(y)
}

glm.score.function <- function(h.transfer, theta, datapoint) {
  # Computes  (yt - h(theta' xt)) * xt = score function
  # for a GLM model with transfer function "h.transfer"
  # 
  # Examples:
  #   normal model : h(x) = x  identity function
  #   poisson model : h(x) = e^x
  #   logistic regression : h(x) = logit(x)
  #
  # Returns: px1 vector of the score (gradient of log-likelihood) 
  xt = datapoint$xt
  yt = datapoint$yt
  # Check dimensions
  #CHECK_EQ(length(xt), length(theta))
  #CHECK_EQ(length(yt), 1, msg="need one-dimensional outcome")
  yt.hat = h.transfer(sum(xt * theta))
  with(datapoint, matrix((yt - yt.hat) * xt, ncol=1))
}

copy.experiment <- function(experiment) {
  e = empty.experiment(experiment$niters)
  for(i in names(experiment)) {
    e[[i]] = experiment[[i]]
  }
  return(e)
}

empty.experiment <- function(niters) {
  # Returns an empty EXPERIMENT object.
  # Useful for initialization and inspection.
  return(list(name="default",
              theta.star=matrix(0, nrow=1, ncol=1),
              niters=niters,
              score.function=function(theta, datapoint) {},
              sample.dataset=function() {}))
}

experiment.description <- function(experiment) {
  # Returns a description of an experiment as a string.
  return(sprintf(" Experiment %s: iters=%d p=%d limit.lr=%.2f",
                 experiment$name, 
                 experiment$niters, experiment$p,
                 experiment$learning.rate(10^9) * 10^9))  
}

experiment.limit.variance <- function(experiment) {
  # Computes the asymptotic variance from the Theorem of (Toulis et al, 2014)
  J = experiment$J
  limit.a = experiment$learning.rate(10^9) * 10^9
  if(limit.a > 10^3) stop("Error. Learning rate grows indefinitely.")
  I = diag(experiment$p)
  V = (limit.a * solve(2 * limit.a * J - I) %*% J)
  #CHECK_TRUE(all(eigen(V)$values >=0), msg="Not valid covariance matrix. Change a")
  
  return(V)
}