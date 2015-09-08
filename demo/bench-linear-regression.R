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
## Unit: milliseconds
##    expr       min        lq      mean    median        uq       max neval
##     sgd  644.8761  676.8018  740.3485  733.2575  776.6705  918.4465    10
##      lm 1467.7145 1566.2102 1648.5728 1608.2134 1773.2074 1816.3088    10
##  glmnet 2392.7664 2437.7863 2636.2133 2600.4307 2789.2635 3091.6138    10
