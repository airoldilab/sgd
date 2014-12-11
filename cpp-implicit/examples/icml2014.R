library(Rcpp)
library(RcppArmadillo)
source("online-algorithms.R")
sourceCpp('implicit.cpp')
sample.covariance.matrix <- function(p) {
  # Samples a low-rank covariance matrix.
  #
  u1 = 0.5 * seq(-1, 1, length.out=p)
  u2 = seq(0.2, 1, length.out=p)
  C = matrix(0, nrow=p, ncol=p)
  diag(C) <- u2
  V =  (C + u1 %*% t(u1))
  #CHECK_TRUE(all(eigen(V)$values > 0))
  V
}

normal.experiment <- function(niters, p=100, lr.scale=1.0) {
  # Normal experiment (linear regression)
  #
  # Assume xt ~ N(0, A)  where A has eigenvalues from 0.01 to 1
  #        yt | xt = xt'θ* + ε ,  where ε ~ N(0, 1) ind.
  #
  # Thus the score function is equal to
  #     (yt - xt'θ) * xt  since h(.) transfer = identity
  #
  # Args:
  #   niters = number of samples (also #iterations for online algorithms)
  #   p = #parameters (dimension of the problem)
  #   lr.scale = scale of learning rate
  # 1. Define θ*
  experiment = empty.experiment(niters)
  experiment$name = "normal"
  # experiment$theta.star = matrix(runif(p, min=0, max=5), ncol=1) 
  experiment$theta.star = matrix(rep(1, p), ncol=1)  # all 1's
  experiment$p = p
  A = sample.covariance.matrix(p)
  experiment$Vx = A
  # 2. Define the sample dataset function.
  experiment$sample.dataset = function() {
    epsilon = matrix(rnorm(niters), ncol=1)
    X = rmvnorm(niters, mean=rep(0, p), sigma=A)
    Y = X %*% experiment$theta.star + epsilon
    if(niters > 1000) {
      # CHECK_MU0(as.vector(Y), 0)
    }
    #CHECK_TRUE(nrow(X) == niters)
    return(list(X=X, Y=Y))
  }
  
  id.fn = function(x) x
  gamma0 = 1 / sum(diag(A))
  lambda0 = min(eigen(A)$values)
  
  # 3. Define the score function
  experiment$h.transfer <- function(u) u
  experiment$score.function = function(theta, datapoint) {
    glm.score.function(h.transfer=id.fn, theta, datapoint)
  }
  
  # 4. Define the learning rate
  experiment$learning.rate <- function(t) {
    # stop("Need to define learning rate per-application.")
    lr.scale * base.learning.rate(t, gamma0=gamma0, alpha=lambda0, c=1)
  }
  
  # 4b. Fisher information
  experiment$J = A
  
  # 5. Define the risk . This is usually the negative log-likelihood
  truth = experiment$theta.star
  experiment$risk <- function(theta) {
    #CHECK_EQ(length(theta), length(truth))
    tmp = 0.5 * t(theta - truth) %*% A %*% (theta - truth)
    #CHECK_EQ(nrow(tmp), 1)
    #CHECK_EQ(ncol(tmp), 1)
    #CHECK_TRUE(all(tmp >= 0))
    return(as.numeric(tmp))
  }
  
  return(experiment)
}

sparse.sample <- function(nsamples, nsize) {
  ## Will sample n elements from 0 to (n-1) such that 
  ## P(X=i) ~ 1 / (1+i)   marginally
  # Creates some pseudo-sparsity.
  s = seq(0, nsize-1)
  return(sample(s, size=nsamples, replace=T, prob=(1+s)^-1))
}


kPoissonQ = 0.2
kThetaStar = log(c(2, 4))
kAlphaRate = 10/3

poisson.experiment <- function(niters) {
  # Poisson regression
  #
  # Assume xt ~ N(0, A)  where A has eigenvalues from 0.01 to 1
  #        yt | xt = Pois(λ) , log(λ) = xt'θ*
  #
  # Thus the score function is equal to
  #     (yt - exp(xt'θ)) * xt
  #
  # Args:
  #   niters = number of samples (also #iterations for online algorithms)
  #   p = #parameters (dimension of the problem)
  #   lr.scale = scale of learning rate
  experiment = empty.experiment(niters)
  experiment$name = "poisson"
  experiment$theta.star = matrix(kThetaStar, ncol=1)  # just a bivariate experiment
  experiment$p = 2
  
  Q = kPoissonQ # probability (0, 1) and (1, 0)
  sample.X <- function(n) {
    # code=0 then x=(0, 0), code=1 x=(1,0) etc.
    code = sample(0:2, size=n, replace=T, prob=c((1-2*Q), Q, Q))
    X = matrix(0, nrow=n, ncol=2)
    X[,1] <- as.numeric(code==1)
    X[,2] <- as.numeric(code==2)
    return(X)
  }
  
  ## 1b Set Covariance of X
  empirical.X = sample.X(20000) ## used for some empirical methods.
  #CHECK_EQ(sum(apply(empirical.X, 1, prod)), 0, msg="No (1,1) vectors")
  #CHECK_MU0(apply(empirical.X, 1, sum)==0, 1-2*Q)
  experiment$Vx = cov(empirical.X)
  
  # 2. Define the sample dataset function.
  experiment$sample.dataset = function() {
    epsilon = matrix(rnorm(niters), ncol=1)
    X = sample.X(niters)
    log.lambdas = X %*% experiment$theta.star
    #CHECK_EQ(length(log.lambdas), niters)
    Y = matrix(rpois(niters, lambda=exp(log.lambdas)), ncol=1)
    #CHECK_TRUE(nrow(X) == niters)
    return(list(X=X, Y=Y))
  }
  
  experiment$h.transfer <- function(u) {
    exp(u)
  }
  # 3. Define the score function
  experiment$score.function = function(theta, datapoint) {
    glm.score.function(h.transfer=exp, theta, datapoint)
  }
  
  
  alpha = kAlphaRate
  # 4. Define the learning rate
  experiment$learning.rate <- function(t) {
    # stop("Need to define learning rate per-application.")
    alpha / t ## note: this will be a bad rate for SGD
              ## min(0.15, alpha/t) fixes it, but the point of this experiment
              ## is to show that the implicit method remains robust across multiple scenarios.
  }
  
  # 4b. Fisher information
  # Recall J= E(h'(theta.star' x) x x')
  experiment$J = matrix(0, nrow=2, ncol=2)
  N = nrow(empirical.X)
  for(i in 1:N) {
    x = empirical.X[i, ]
    h.prime = exp(sum(experiment$theta.star * x))
    experiment$J <- experiment$J + h.prime * x %*% t(x)
  }
  experiment$J = experiment$J / N
  #CHECK_NEAR(diag(experiment$J), Q * exp(experiment$theta.star), tol=0.05)
  # 5. Define the risk . This is usually the negative log-likelihood
  truth = experiment$theta.star
  experiment$risk <- function(theta) {
    vector.dist(theta, experiment$theta.star)
  }
  
  return(experiment)
}




run.poisson.experiment <- function(niters=100, nsamples=20) {
  e = poisson.experiment(niters=niters)
  V = experiment.limit.variance(e)
  ## 1. Theoretical values
  Var.numerical = experiment.limit.variance(e)
  limit.a = e$learning.rate(10^9) * 10^9
  #CHECK_NEAR(limit.a, kAlphaRate, 0.01)
  gamma = 2 * limit.a * kPoissonQ
  print(sprintf("Limit a=%.3f  gamma=%.3f", limit.a, gamma))
  Var.theoretical = (gamma/2) * diag(c(exp(e$theta.star) / (gamma * exp(e$theta.star)-1)))
  print("Theoretical variance")
  print(Var.theoretical)
  print("Numerical variance")
  print(Var.numerical)
  
  ## 2. Run all algorithms.
  out = run.online.algorithm.many(e, algorithm.names=c(kSGD, kIMPLICIT), nsamples=nsamples)
  
  # 3. Get estimates
  theta.im = t(out$implicit.onlineAlgorithm[[niters]]) # samples x p
  theta.sgd = t(out$sgd.onlineAlgorithm[[niters]]) # samples x p
  
  ## 4. Variance of Implicit method
  Var.implicit = (1/e$learning.rate(niters)) * cov(theta.im)
  print("Variance of implicit updates")
  print(Var.implicit)
  
  ## 5. Variance of SGD method
  #   5b. Find bad estimates in SGD
  sgd.risk = sapply(1:nrow(theta.sgd), function(i) e$risk(theta.sgd[i,]))
  im.risk =  sapply(1:nrow(theta.im), function(i) e$risk(theta.im[i,]))
  bad.estimates = which(sgd.risk > quantile(sgd.risk, probs=0.75))
  theta.sgd = theta.sgd[-bad.estimates,] ## remove bad estimates
  print(sprintf("--->  There are %d bad estimates (>Q3) out of %d in total, for SGD ", 
                length(bad.estimates),
                nsamples))
  Var.sgd = (1/e$learning.rate(niters)) * cov(theta.sgd)
  print("Variance of SGD updates (< Q3)")
  print(Var.sgd)
  
  ## 6. create quantiles 
  print("Risk quantiles for SGD and Implicit")
  print(round(quantile(sqrt(2) * sgd.risk, probs=c(0.25, 0.5, 0.75, 0.85, 0.95, 1.0)), 2))
  print(round(quantile(sqrt(2) * im.risk, probs=c(0.25, 0.5, 0.75, 0.85, 0.95, 1.0)), 2))
  
  ## 7. Save results 
  save(out, file="out/poisson-experiment.Rdata")
}

postProcess.poisson <- function() {
  load("out/poisson-experiment.Rdata")
  niters = length(out$sgd.onlineAlgorithm)
  nsamples = ncol(out$sgd.onlineAlgorithm[[1]])
  print(sprintf("iters = %d ,samples=%d", niters, nsamples))
  e = poisson.experiment(niters=niters)
  limit.a = e$learning.rate(10^9) * 10^9
  gamma = 2 * limit.a * kPoissonQ
  
  Var.theoretical = (gamma/2) * diag(c(exp(e$theta.star) / (gamma * exp(e$theta.star)-1)))
  
  ts = seq(1, niters)
  y = sapply(ts, function(t) { theta.im = t(out$implicit.onlineAlgorithm[[t]]);
                               V = (1/e$learning.rate(t)) * cov(theta.im);
                               matrix.dist(V, Var.theoretical)
  })
  plot(ts, y, type="l")
}




