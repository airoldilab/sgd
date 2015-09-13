#!/usr/bin/env Rscript
# This is used to benchmark glmnet for 5e4 observations and 1e4 dimensions.

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
nSim <- 1    # number of runs
N <- 5e4      # size of minibatch
d <- 1e4
rho <- 0

times.glmnet <- rep(0, nSim)

set.seed(42)
for (i in 1:nSim) {
  print(sprintf("Running simulation %i of %i...", i, nSim))
  # Generate stream of data.
  X <- genx(N, d, rho)
  theta <- ((-1)^(1:d))*exp(-2*((1:d)-1)/20)
  eps <- rnorm(N)
  k <- 3
  y <- X %*% theta + k * eps

  # glmnet doesn't work on streaming data
  time_start <- proc.time()[3]
  glmnet.theta <- glmnet(X, y, alpha=1, standardize=FALSE,
    type.gaussian="covariance")
  times.glmnet[i] <- as.numeric(proc.time()[3] - time_start)
}
print(mean(times.glmnet))

save(times.glmnet, glmnet.theta, file=sprintf("out/linear-regression-glmnet.RData"))
