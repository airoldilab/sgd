#!/usr/bin/env Rscript
# Demo usage of sgd for linear regression on simulated normal data.
#
# Data generating process:
#   Y = Binomial(N, p), where
#     p (see generate.data)
#     X ~ correlated Normal (see genx)
#
# Dimensions:
#   N=1e3 observations
#   d=5 parameters

# Function taken from Friedman et al.
genx = function(n,p,rho){
  #    generate x's multivariate normal with equal corr rho
  # Xi = b Z + Wi, and Z, Wi are independent normal.
  # Then Var(Xi) = b^2 + 1
  #  Cov(Xi, Xj) = b^2  and so cor(Xi, Xj) = b^2 / (1+b^2) = rho
  z=rnorm(n)
  if(abs(rho)<1){
    beta=sqrt(rho/(1-rho))
    x0=matrix(rnorm(n*p),ncol=p)
    A = matrix(z, nrow=n, ncol=p, byrow=F)
    x= beta * A + x0
  }
  if(abs(rho)==1){ x=matrix(z,nrow=n,ncol=p,byrow=F)}

  return(x)
}

generate.data <- function(n, p, rho=0.2) {
  ## Generate data
  #   #
  #   # Returns:
  #   #   LIST(X, Y, censor, true.beta)
  #   #     X = Nxp matrix of covariates.
  #   #     Y = Nx1 vector of observed times.
  #   #     censor = Nx1 vector {0, 1} of censor indicators.
  #   #     true.beta = p-vector of true model parameters.
  #   #     M = (Y, censor) as matrix

  X = genx(n, p, rho=rho)
  # rates.
  # rates = runif(n, min=1e-2, max=10)
  # Y = rexp(n,
  # beta = solve(t(X) %*% X) %*% t(X) %*% log(rates)
  beta = 10  *((-1)^(1:p))*exp(-2*((1:p)-1)/20)
  # beta = 10 * seq(1, p)**(-0.5)
  # warning("Large coefficients")
  pred = exp(X %*% beta)
  Y = rexp(n, rate =pred)

  q3 = quantile(Y, prob=c(0.8))  # Q3 of Y
  epsilon = 0.001 # probability of censoring smallest Y
  k = log(1/epsilon - 1) / (q3 - min(Y))
  censor.prob = (1 + exp(-k * (Y-q3)))**(-1)

  C = rbinom(n, size=1, prob= censor.prob)

  ## Order the data
  order.i = order(Y)
  X = X[order.i, ]
  Y = Y[order.i]
  C = C[order.i]

  M = matrix(0, nrow=n, ncol=2)
  colnames(M) <- c("time", "status")
  M[, 1] <- Y
  M[, 2] <- 1-C
  return(list(X=X, Y=Y, censor=C, M=M, true.beta=beta))
}

library(sgd)

# Dimensions
N <- 1e3
d <- 5

# Generate data.
set.seed(42)
data <- generate.data(N, d)
X <- data$X
y <- 1 - data$censor # y=1 if fail, 0 if censor
dat <- data.frame(y=y, x=X)

t <- data$Y # times of observations
theta <- data$true.beta # true coefficients

# Explicit
sgd.theta <- sgd(y ~ . - 1, data=dat, model="cox",
                 sgd=list(method="sgd", lr="adagrad", npasses=5))
sgd.theta$coefficients

# Implicit
sgd.theta <- sgd(y ~ . - 1, data=dat, model="cox",
                 sgd=list(method="implicit",
                          lr="adagrad",
                          lr.control=c(0.1, NA),
                          npasses=5))
sgd.theta$coefficients
