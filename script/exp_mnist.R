# Run logistic regression on MNIST dataset
# The dataset can be downloaded from
#   http://yann.lecun.com/exdb/mnist/
# To run this script, the working directory should be the base repo
#   data files should be stored in "data/"

source("script/load_mnist.R")
source("script/multilogit.R")
source("script/multiplot.R")
source("script/plot.R")

library(sgd)

dat <- load_mnist()
X <- dat$train$x
y <- dat$train$y
X_test <- dat$test$x
y_test <- dat$test$y

methods <- list("sgd", "ai-sgd", "implicit")
lrs <- list("one-dim", "one-dim", "one-dim")
np <- list(1, 1, 1)
names <- methods

out <- run_exp(methods, names, lrs, np, X, y, X_test, y_test)
do.call("multiplot", out)
