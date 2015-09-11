#!/usr/bin/env Rscript
# Benchmark sgd package for linear regression on simulated data from a
# correlated normal distribution. This follows the experiment in Section 5.1 of
# Friedman et al. (2010).
#
# Data generating process:
#   Y = sum_{j=1}^p X_j*β_j + k*ɛ, where
#     X ~ Multivariate normal where each covariate Xj, Xj' has equal correlation
#       ρ; ρ ranges over (0,0.1,0.2,0.5,0.9,0.95) for each pair (n, d)
#     β_j = (-1)^j exp(-2(j-1)/20)
#     ɛ ~ Normal(0,1)
#     k = 3
#
# Dimensions:
#   n=1000, d=100
#   n=5000, d=100
#   n=100, d=1000
#   n=100, d=5000
#   n=100, d=20000
#   n=100, d=50000

library(sgd)
library(glmnet)
library(microbenchmark)

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

# Dimensions
Ns <- c(1000, 5000, 100, 100, 100, 100)
ds <- c(100, 100, 1000, 5000, 20000, 50000)
rhos <- c(0, 0.1, 0.2, 0.5, 0.9, 0.95)

# explicit sgd will error the loop for higher correlation
rhos <- c(0, 0.1, 0.2)

set.seed(42)
benchmark <- list()
idx <- 1
for (i in 1:length(Ns)) {
  for (j in 1:length(rhos)) {
    N <- Ns[i]
    d <- ds[i]
    rho <- rhos[j]
    # Generate data.
    X <- genx(N, d, rho)
    theta <- ((-1)^(1:d))*exp(-2*((1:d)-1)/20)
    eps <- rnorm(N)
    k <- 3
    y <- X %*% theta + k * eps

    # Benchmark!
    benchmark[[idx]] <- microbenchmark(
      aisgd=sgd(X, y, data=dat, model="lm",
              sgd.control=list(method="ai-sgd", npasses=1, pass=T)),
      sgd=sgd(X, y, data=dat, model="lm",
              sgd.control=list(method="sgd", npasses=1, pass=T)),
      glmnet=glmnet(X, y, alpha=1, standardize=FALSE, type.gaussian="covariance"),
      times=10L, unit="s"
    )
    names(benchmark)[idx] <- sprintf("N: %i; d: %i; rho: %0.2f", N, d, rho)
    idx <- idx + 1
  }
}
