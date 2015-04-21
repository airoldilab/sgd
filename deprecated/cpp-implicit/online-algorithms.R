# Copyright (c) 2013
# Panos Toulis, ptoulis@fas.harvard.edu
#
# Online algorithms.
source("terminology.R")
source("experiment.R")

run.online.algorithm.many <- function(experiment,
                                      algorithm.names=kImplementedOnlineAlgorithms, 
                                      nsamples) {
  # Will generate many estimates for the algorithm
  #
  # The result will be a LIST of the form
  #   out[[algoName]][[#sample]]
  # e.g. out[[sgd.onlineAlgorithm]][[5]] = 5th-sample (OnlineOutput object)
  # So out[[sgd]][[5]] has #nsamples of θ5
  algo.fn = onlineAlgorithm.wrapper(algorithm.names)
  # initialize
  out.all = list()
  for(algoName in algorithm.names) {
    out.all[[algoName]] <- list()
    for(t in 1:experiment$niters) {
      out.all[[algoName]][[t]] = matrix(0, nrow=experiment$p, ncol=nsamples)
    }
  }
  pb <- txtProgressBar(style=3)
  for(i in 1:nsamples) {
    dataset = experiment$sample.dataset()
    for(algoName in names(algo.fn)) {
      out.tmp = run.online.algorithm(dataset, experiment, algo.fn[[algoName]])
      #CHECK_EQ(ncol(out.tmp$estimates), experiment$niters, msg="Correct #iters")
      for(t in 1:experiment$niters) {
        out.all[[algoName]][[t]][, i] <- out.tmp$estimates[, t]
      }
    }
    setTxtProgressBar(pb, value=i/nsamples)
  }
  #CHECK_multipleOnlineOutput(out.all, experiment=experiment)
  return(out.all)
}

# Convention is that every online algorithm should be named:
#
# X.onlineAlgorithm, where X = name of the algorithm
run.online.algorithm <- function(dataset, experiment, algorithm, verbose=F) {
  # Runs the specific online learning algorithm
  # See terminology for the definition of OnlineAlgorithm
  #
  #CHECK_dataset(dataset)
  #CHECK_experiment(experiment)
  # Will return the "out" object (of type OnlineOutput)
  out = empty.onlineOutput(dataset)
  nsamples = dataset.size(dataset)$nsamples
  #CHECK_EQ(nsamples, experiment$niters)  # iterations = samples
  pb = NA
  algo.name = as.character(substitute(algorithm))
  if(verbose) {
    cat(sprintf("Running algorithm %s, Experiment=%s, samples=%d \n",
                algo.name, experiment$name, experiment$niters))
    pb <- txtProgressBar(style=3)
  }
  
  for(t in 1:nsamples) {
    ## Run all iterations here.
    # History has data from 1 to t
    # Recall that theta_t = estimate of theta AFTER seeing datapoint t.
    history = list(X=matrix(dataset$X[1:t, ], ncol=experiment$p),
                   Y=matrix(dataset$Y[1:t, ], ncol=1))
    # 1. Runs the online-algorithm step.
    theta.new = algorithm(t, online.out=out,
                          data.history=history,
                          experiment=experiment)
    out <- add.estimate.onlineOutput(out, t, estimate=theta.new)
    if(verbose)
      setTxtProgressBar(pb, value=t/nsamples)
  }
  
  # If ASGD we need to average all the estimates.
  if(length(grep("asgd", algo.name)) > 0) {
    out <- asgd.transformOnlineOutput(out)
  }
  out$last = out$estimates[, nsamples]
  #CHECK_onlineOutput(out)
  return(out)
}

sgd.onlineAlgorithm <- function(t, online.out, data.history, experiment) {
  # Implements the SGD algorithm
  #
  datapoint = get.dataset.point(dataset=data.history, t=t)
  at = experiment$learning.rate(t)
  theta.old = onlineOutput.estimate(online.out, t-1)
  score.t = experiment$score.function(theta.old, datapoint)
  theta.new = theta.old + at * score.t
  return(theta.new)
}

asgd.onlineAlgorithm <- function(t, online.out, data.history, experiment) {
  # Implements the ASGD algorithm (Polyak 1992)
  # This will perform the SGD updates but...
  # We add an expection in run.onlineAlgorithm so that
  # the ASGD will average over the SGD estimates.
  return(sgd.onlineAlgorithm(t, online.out, data.history, experiment))
}

asgd.transform.output <- function(sgd.onlineOutput) {
  ## This is ASGD need to rework the estimates
  out = sgd.onlineOutput
  estimates = out$estimates
  avg.estimates = matrix(0, nrow=nrow(estimates), ncol(estimates))
  avg.estimates[,1] = estimates[,1]
  for(t in 2:ncol(estimates)) {
    avg.estimates[,t] = (1-1/t) * avg.estimates[,t-1] + (1/t) * estimates[,t]
  }
  out$estimates = avg.estimates
  return(out)
}

oracle.onlineAlgorithm <- function(t, online.out, data.history, experiment) {
  return(experiment$theta.star)
}

implicit.onlineAlgorithm <- function(t, online.out, data.history, experiment) {
  datapoint = get.dataset.point(dataset=data.history, t=t)
  at = experiment$learning.rate(t)
  xt = datapoint$xt
  norm.xt = sum(xt^2)
  yt = datapoint$yt
  theta.old = onlineOutput.estimate(online.out, t-1)
  get.score.coeff <- function(ksi) {
    # this returns the value  yt - h(theta_{t-1}' xt + xt^2 ξ)  -- for a GLM
    # this is a scalar.
    return(yt - experiment$h.transfer(sum(theta.old * xt) + norm.xt * ksi))
  }
  # 1. Define the search interval
  rt = at * get.score.coeff(0)
  Bt = c(0, rt)
  if(rt < 0) {
    Bt <- c(rt, 0)
  }
  
  implicit.fn <- function(u) {
    u  - at * get.score.coeff(u)
  }
  # 2. Solve implicit equation
  xit = NA
  if(Bt[2] != Bt[1])
    xit = uniroot(implicit.fn, interval=Bt)$root
  else 
    xit = Bt[1]
  theta.new = theta.old + xit * xt
  return(theta.new)
}


kImplementedOnlineAlgorithms <<- ls()[grep("\\.onlineAlgorithm", ls())]
onlineAlgorithm.wrapper <- function(algo.names) {
  # Given algorithm names it will return the functions
  # that implement them.
  # Can be used then in run.online.algorithm
  #
  fn.list = list("sgd.onlineAlgorithm"=sgd.onlineAlgorithm,
                 "implicit.onlineAlgorithm"=implicit.onlineAlgorithm,
                 "asgd.onlineAlgorithm"=asgd.onlineAlgorithm,
                 "oracle.onlineAlgorithm"=oracle.onlineAlgorithm)
  #CHECK_MEMBER(algo.names, kImplementedOnlineAlgorithms)
  #CHECK_EQ(length(kImplementedOnlineAlgorithms), length(names(fn.list)))
  ret.list = list()
  for(algoName in algo.names) {
    ret.list[[algoName]] = fn.list[[algoName]]
  }
  return(ret.list)
}

