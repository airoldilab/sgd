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
##     sgd 1.122968 1.251983 1.369997 1.296095 1.526145 1.786849    10
##      lm 1.546782 1.623304 1.838424 1.795919 2.061442 2.201741    10
##  glmnet 2.452500 2.584168 2.682394 2.627161 2.735530 3.093342    10
