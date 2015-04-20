# Copyright (c) 2013
# Panos Toulis, ptoulis@fas.harvard.edu
#
rm(list=ls())
#source("../r-toolkit/checks.R")
#source("../r-toolkit/logs.R")

# For convenience.
kSGD = "sgd.onlineAlgorithm"
kIMPLICIT = "implicit.onlineAlgorithm"
kMainAlgos = c(kSGD, kIMPLICIT)
kAlgoHumanNames <- function(algo) {
  if(length(grep("sgd", algo)))
    return("SGD")
  else if(length(grep("implicit", algo))) {
    return("Implicit")
  }
}

## For information about the terminology please check the 
## README file locally or in address https://github.com/ptoulis/implicit-glms

get.dataset.point <- function(dataset, t) {
  #CHECK_TRUE(nrow(dataset$X) >= t && nrow(dataset$Y) >= t)
  return(list(xt=dataset$X[t, ],
              yt=dataset$Y[t, ]))
}

dataset.size <- function(dataset) {
  # Return a LIST with the #samples and the #parameters
  return(list(nsamples=nrow(dataset$X),
              p=ncol(dataset$X)))
}

empty.onlineOutput <- function(dataset)  {
  size = dataset.size(dataset)  
  return(list(estimates=matrix(0, nrow=size$p, ncol=size$nsamples)))
}

add.estimate.onlineOutput <- function(out, t, estimate) {
  # Adds an estimate in the OnlineOutput object
  #CHECK_TRUE(t <= ncol(out$estimates), msg="t < #total samples")
  #CHECK_EQ(length(estimate), nrow(out$estimates), msg="Correct p=#parameters")
  out$estimates[, t] <- estimate
  return(out)
}

onlineOutput.estimate <- function(out, t) {
  if(t==0) {
    # warning("Default is to return 0-vector, for t=0")
    return(matrix(0, nrow=nrow(out$estimates), ncol=1))
  }
  #CHECK_TRUE(t <= ncol(out$estimates), msg="t < #total samples")
  return(matrix(out$estimates[, t], ncol=1))
}

onlineOutput.risk <- function(out, experiment) {
  apply(out$estimates, 2, experiment$risk)
}

# Functions for the MultipleOnlineOutput object.
CHECK_multipleOnlineOutput <- function(mu.out, experiment) {
  if(experiment$niters == 0) return;
  # get the first algo
  mu.out = mu.out[[names(mu.out)[1]]]
  #CHECK_EQ(length(mu.out), experiment$niters)
  random.i = sample(1:experiment$niters, 1)
  #CHECK_EQ(nrow(mu.out[[random.i]]), experiment$p)
}

## Some L2 distance functions.
vector.dist <- function(x1, x2) {
  # ||x1-x2|| / sqrt(n)  == normalized by the size.
  m1 = matrix(x1, ncol=1)
  m2 = matrix(x2, ncol=1)
  return(matrix.dist(m1, m2))
}

matrix.dist <- function(m1, m2) {
  #CHECK_EQ(nrow(m1), nrow(m2))
  #CHECK_EQ(ncol(m1), ncol(m2))
  sqrt(sum((m1-m2) %*% t(m1-m2))) / sqrt(nrow(m1) * ncol(m1))
}

default.var.dist = function(experiment, nsamples) {
  force(experiment)
  Sigma.theoretical = limit.variance(experiment)
  
  dist.fn <- function(theta.matrix, t) {
    #CHECK_EQ(nrow(theta.matrix), experiment$p)
    #CHECK_EQ(ncol(theta.matrix), nsamples)
    C = cov(t(theta.matrix))
    matrix.dist(t * C, Sigma.theoretical)
  }
}

default.bias.dist <- function(experiment) {
  dist.fn <- function(theta, t) {
    return(vector.dist(experiment$theta.star, theta))
  }
}

mul.OnlineOutput.vapply <- function(mul.out, experiment, algo, theta.t.fn, summary.fn) {
  # Applies a function to a MultipleOnlineOutput object.
  #
  # Returns a vector V = (niters x 1) vector of values where
  #  V(t) = summary(fn(theta_t1), fn(thetat2), ...)
  #
  # theta_tj = j-th sample of theta_t
  # summary = summary function, e.g. specific quantile, max etc.
  #
  #CHECK_MEMBER(algo, names(mul.out))
  out = mul.out[[algo]]
  maxT = length(out)
  #CHECK_EQ(experiment$niters, maxT)
  #CHECK_EQ(experiment$p, nrow(out[[1]]))
  # out = LIST(1..niters) of (p x n) matrices.
  sapply(1:maxT, function(t) {
    thetat.samples = out[[t]]
    # For each sample, compress the parameter vector to one value.
    one.theta = apply(thetat.samples, 2, function(x) theta.t.fn(x, t))
    # one.theta = vector (samples x 1)
    # Summarize the sample vector.
    summary.fn(one.theta)
  })
}

mul.OnlineOutput.mapply <- function(mul.out, experiment, algo, fn) {
  # Applies a function to a MultipleOnlineOutput object.
  # 
  # Args:
  #   experiment = the EXPERIMENT object (see terminology)
  #   mul.out = MultipleOnlineOutput object.
  #   algo = name of the algorithm (character)
  #   fn = function (p x n) -> R
  #
  # Returns a vector V = (niters x 1) vector of values where
  #   V = (fn(out_1), fn(out_2), ...)
  #   V(t)  = fn(out_t) Here fn() accepts a matric.
  #CHECK_MEMBER(algo, names(mul.out))
  algoOut = mul.out[[algo]]  # LIST[[t]] = matrix(p x n)
  maxT = length(algoOut)
  #CHECK_EQ(experiment$niters, maxT)
  #CHECK_EQ(experiment$p, nrow(algoOut[[1]]))
  res = fn(algoOut[[1]], 1)
  #CHECK_TRUE(is.null(dim(res)) || length(res) == 1, msg="Output should be 1-dim")
  sapply(1:maxT, function(t) {
    thetat.samples = algoOut[[t]]
    fn(thetat.samples, t)
  })
}

benchmark.nsamples <- function(benchmark) {
  algo1 = names(benchmark$mulOut)[1]
  return(ncol(benchmark$mulOut[[algo1]][[1]]))
}

benchmark.niters <- function(benchmark) {
  algo = benchmark.algos(benchmark)[1]
  return(length(benchmark$mulOut[[algo]]))
}

benchmark.theta.samples <- function(benchmark, algoName, t) {
  x  = benchmark$mulOut[[algoName]][[t]]
  #CHECK_TRUE(!is.null(x))
  return(x)
}

benchmark.algos <- function(benchmark) {
  return(names(benchmark$mulOut))
}

benchmark.algo.low <- function(benchmark, algoName) {
  x = as.numeric(benchmark$lowHigh[[algoName]]$low)
  #CHECK_TRUE(!is.null(x))
  return(x)
}

benchmark.algo.high <- function(benchmark, algoName) {
  x = as.numeric(benchmark$lowHigh[[algoName]]$high)
  CHECK_TRUE(!is.null(x))
  return(x)
}


CHECK_benchmark <- function(benchmark) {
  #CHECK_MEMBER(names(benchmark), c("mulOut", "lowHigh", "experiment", "draw", "name"))
  #CHECK_MEMBER(names(benchmark$lowHigh), kImplementedOnlineAlgorithms)
  #CHECK_multipleOnlineOutput(benchmark$mulOut, experiment=benchmark$experiment)
  
  for(algo in names(benchmark$lowHigh)) {
    #CHECK_SETEQ(names(benchmark$lowHigh[[algo]]), c("low", "high"))
    for(i in names(benchmark[[algo]])) {
      #CHECK_numeric(benchmark[[algo]][[i]])
      #CHECK_TRUE(is.vector(benchmark[[algo]][[i]]))
    }
  }
  algos = benchmark.algos(benchmark)
  ralgo = sample(algos, 1)
  # 0. Same algos.
  #CHECK_SETEQ(names(benchmark$lowHigh), algos)
  # 1. check consistency in lengths (niters)
  #CHECK_EQ(benchmark$experiment$niters, length(benchmark$mulOut[[ralgo]]))
  # 2. Check consistency in "p" (#params)
  #CHECK_EQ(benchmark$experiment$p, nrow(benchmark.theta.samples(benchmark, ralgo, 1)))
  # 3. Check consistency in "nsamples"
  for(algo in names(benchmark$mulOut)) {
    niters = benchmark.niters(benchmark)
    random.t = sample(1:niters, 1)
    #CHECK_TRUE(ncol(benchmark.theta.samples(benchmark, algoName=algo, random.t)),
    #           ncol(benchmark.theta.samples(benchmark, algo, 1)))
  }
}

# CHECKS for definition of *params objects.
CHECK_processParams <- function(procParams) {
  #CHECK_MEMBER(names(procParams), c("vapply", "name"))
  #CHECK_TRUE(is.logical(procParams$vapply))
}

CHECK_mulOutParams <- function(moutParams) {
  #CHECK_MEMBER(names(moutParams), c("algos", "experiment", "nsamples"))
  #CHECK_experiment(moutParams$experiment)
  #CHECK_MEMBER(moutParams$algos, kImplementedOnlineAlgorithms)
  #CHECK_numeric(moutParams$nsamples)
}

CHECK_dataset <- function(dataset) {
  #CHECK_SETEQ(names(dataset), c("X", "Y"))
  #CHECK_columnVector(dataset$Y)
  #CHECK_EQ(nrow(dataset$X), nrow(dataset$Y))
  #CHECK_numeric(dataset$X)
  #CHECK_numeric(dataset$Y)
}

CHECK_onlineAlgorithm = function(algorihm) {
  #CHECK_SETEQ(names(formals(algorithm)), c("t", "data.history", "experiment", "online.out"))
}

CHECK_columnVector <- function(x) {
  #CHECK_GE(nrow(x), 1, msg="Has >=1 row")
  #CHECK_EQ(ncol(x), 1, msg="Only one col")
}

CHECK_rowVector <- function(x) {
  #CHECK_columnVector(t(x))
}

CHECK_numeric <- function(x) {
  # Checks that there are no NA's, Inf's etc
  #CHECK_TRUE(all(!is.na(x)), msg="No NAs")
  #CHECK_TRUE(all(!is.infinite(x)), msg="No Infs")
  #CHECK_TRUE(is.numeric(x), msg="Should be numeric")
}

CHECK_experiment <- function(experiment) {
  #CHECK_MEMBER(names(experiment),
  #             c("name", "p", "theta.star", "niters", 
  #               "sample.dataset", "Vx", "J",
  #               "h.transfer", "score.function",
  #               "learning.rate", "scale",
  #               "risk"), msg="Correct fields for the experiment")
  #CHECK_columnVector(experiment$theta.star)
  #CHECK_EQ(experiment$p, length(experiment$theta.star))
  D = experiment$sample.dataset()
  #CHECK_dataset(D)
  point = get.dataset.point(D, 2)
  #CHECK_SETEQ(names(formals(experiment$score.function)),
  #            c("theta", "datapoint"))
  p = length(experiment$theta.star)
  bad.theta = rep(0, length(experiment$theta.star) - 1)
  # we send a theta with wrong dimension
  #CHECK_EXCEPTION(experiment$score.function(bad.theta, point))
  #CHECK_columnVector(experiment$score.function(experiment$theta.star, point))
  
  #CHECK_MEMBER("t", names(formals(experiment$learning.rate)))
  #CHECK_EQ(experiment$risk(experiment$theta.star), 0, msg="Risk of ground-truth is 0")
  for(i in 1:100) {
    theta.random = runif(length(experiment$theta.star), min=-2, max=2)
    #CHECK_TRUE(experiment$risk(theta.random) >= 0, msg="Loss has to be positive.")
  }
  # check that the covariance matrix is positive definite
  #CHECK_TRUE(all(eigen(experiment$Vx)$values > 0), msg="Cov matrix is nonneg def")
}

CHECK_onlineOutput <- function(onlineOut) {
  #CHECK_MEMBER(names(onlineOut), c("estimates", "last"))
  #CHECK_numeric(onlineOut$estimates)
  #CHECK_EQ(onlineOut$estimates[ , ncol(onlineOut$estimates)], onlineOut$last)
}