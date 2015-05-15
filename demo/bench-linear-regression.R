#!/usr/bin/env Rscript
# Benchmark sgd package for linear regression on simulated data from a normal
# distribution.
#
# Data generating process:
#   Y = X %*% theta + epsilon, where
#     X ~ Normal(0, 1)
#     theta = (5,...,5)
#     epsilon ~ Normal(0,1)
#
# Dimensions:
#   N=1e5 observations
#   d=1e2 parameters

library(sgd)
library(glmnet)
library(microbenchmark)

# Dimensions
N <- 1e5
d <- 1e2

# Generate data.
set.seed(42)
X <- matrix(rnorm(N*d), ncol=d)
theta <- rep(5, d+1)
eps <- rnorm(N)
y <- cbind(1, X) %*% theta + eps
dat <- data.frame(y=y, x=X)

# Benchmark!
benchmark <- microbenchmark(
  sgd=sgd(y ~ ., data=dat, model="lm",
          sgd.control=list(method="implicit")),
  lm=lm(y ~ ., data=dat),
  glmnet=glmnet(X, y, alpha=1, standardize=FALSE, type.gaussian="naive"),
  times=10L
)
benchmark
## Output (for 2.6 GHz, Intel Core i5)
## Unit: seconds
##    expr      min       lq     mean   median       uq      max neval
##     sgd 1.302610 1.393843 1.456990 1.459721 1.483716 1.709765    10
##      lm 1.650819 1.706835 1.776254 1.763900 1.786094 1.938011    10
##  glmnet 2.451650 2.552243 2.617811 2.623174 2.635034 2.863428    10
