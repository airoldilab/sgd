# Run logistic regression on  MNIST dataset
# The dataset can be downloaded from 
#   http://yann.lecun.com/exdb/mnist/
# To run this script, the working directory should be "script"
#   data files should be stored in "script/data"

source("load_mnist.R")
source("multilogit.R")
source("plot.R")

library(sgd)

dat <- load_mnist()
X <- dat$train$x
y <- dat$train$y
X_test <- dat$test$x
y_test <- dat$test$y


methods <- list("sgd", "ai-sgd", "implicit")
lrs <- list("adagrad", "one-dim", "one-dim")
np <- list(3, 3, 3)
names <- methods

methods <- list("ai-sgd")
lrs <- list("one-dim")
np <- list(4)
names <- methods 
run_exp(methods, names, lrs, np, X, y, X_test, y_test)
