#!/usr/bin/env Rscript
# Benchmark sgd package for linear regression on simulated data from a
# correlated normal distribution. This follows the experiment in Section 5.1 of
# Friedman et al. (2010).
#
# Data generating process:
#   Y = sum_{j=1}^p X_j*β_j + k*ɛ, where
#     X ~ Multivariate normal where each covariate Xj, Xj' has equal correlation
#       ρ; ρ ranges over (0,0.1,0.2,0.5,0.9,0.95) for each pair (n, p)
#     β_j = (-1)^j exp(-2(j-1)/20)
#     ɛ ~ Normal(0,1)
#     k = 3
#
# Dimensions:
#   n=1,000, d=100
#   n=10,000, d=1,000
#   n=100,000, d=10,000
#   n=1,000,000, d=50,000
#   n=10,000,000, d=100,000

library(sgd)
library(glmnet)

# Function taken from Friedman et al.
genx = function(n,p,rho){
  #    generate x's multivariate normal with equal corr rho
  # Xi = b Z + Wi, and Z, Wi are independent normal.
  # Then Var(Xi) = b^2 + 1
  #  Cov(Xi, Xj) = b^2  and so cor(Xi, Xj) = b^2 / (1+b^2) = rho
  z=rnorm(n)
  if(abs(rho)<1){
    beta=sqrt(rho/(1-rho))
    x=matrix(rnorm(n*p),ncol=p)
    A = matrix(rnorm(n), nrow=n, ncol=p, byrow=F)
    x= beta * A + x
  }
  if(abs(rho)==1){ x=matrix(rnorm(n),nrow=n,ncol=p,byrow=F)}

  return(x)
}

# Dimensions: Put them manually here!
nSim <- 10    # number of runs
N <- 1e4      # size of minibatch
nstreams <- 1 # number of streams
d <- 1e3
rho <- 0

times.aisgd <- rep(0, nSim)
times.sgd <- rep(0, nSim)
times.glmnet <- rep(0, nSim)
converged.aisgd <- FALSE
converged.sgd <- FALSE
converged.glmnet <- FALSE

set.seed(42)
for (i in 1:nSim) {
  for (nstream in 1:nstreams) {
    # Generate stream of data.
    X <- genx(N, d, rho)
    theta <- ((-1)^(1:d))*exp(-2*((1:d)-1)/20)
    eps <- rnorm(N)
    k <- 3
    y <- X %*% theta + k * eps

    # AI-SGD
    if (!converged.aisgd) {
      if (nstream == 1) {
        start <- rnorm(d, mean=0, sd=1e-5)
      } else {
        start <- aisgd.theta$coefficients
      }
      aisgd.theta <- sgd(X, y, model="lm",
        sgd.control=list(method="ai-sgd", npasses=1, start=start))
      times.aisgd[i] <- times.aisgd[i] + max(aisgd.theta$times)
      if (aisgd.theta$converged) {
        converged.aisgd <- TRUE
      }
    }

    # explicit SGD
    if (!converged.sgd) {
      if (nstream == 1) {
        start <- rnorm(d, mean=0, sd=1e-5)
      } else {
        start <- sgd.theta$coefficients
      }
      sgd.theta <- sgd(X, y, model="lm",
              sgd.control=list(method="sgd", npasses=1, start=start))
      times.sgd[i] <- times.sgd[i] + max(sgd.theta$times)
      if (sgd.theta$converged) {
        converged.sgd <- TRUE
      }
    }

    # glmnet doesn't work on streaming data
    if (nstreams == 1) {
      time_start <- proc.time()[3]
      glmnet.theta <- glmnet(X, y, alpha=1, standardize=FALSE,
        type.gaussian="covariance")
      times.glmnet[i] <- as.numeric(proc.time()[3] - time_start)
    }
  }
}
print(mean(times.aisgd * 10))
print(mean(times.sgd * 10))
print(mean(times.glmnet))

# Tweaks:
# * For 100 lambda values, I simply just do one lambda value and multiply the time
# by 10.
# * The times outputted from sgd only includes the C++ time.
# * TODO detect convergence
