# Run logistic regression on MNIST dataset
# The dataset can be downloaded from
#   http://yann.lecun.com/exdb/mnist/
# To run this script, the working directory should be the base repo
#   data files should be stored in "data/"

source("script/load_mnist.R")
source("script/plot.R")
source("script/run_exp.R")

library(sgd)
library(gridExtra)

dat <- load_mnist()
X_train <- dat$train$x
y_train <- dat$train$y
X_test <- dat$test$x
y_test <- dat$test$y

# Set task to be binary classification on digit 9.
y_train[y_train != 9] <- 0
y_train[y_train == 9] <- 1
y_test[y_test != 9] <- 0
y_test[y_test == 9] <- 1

methods <- list("sgd", "implicit", "sgd", "ai-sgd")
lrs <- list("one-dim", "one-dim", "adagrad", "one-dim")
np <- list(1, 1, 1, 1)
names <- list("sgd", "implicit", "adagrad", "ai-sgd")
dataset <- "mnist"

methods <- list("implicit")
lrs <- list("one-dim")
np <- list(1)
names <- list("implicit")
dataset <- "mnist"

out_mnist <- run_exp(methods, names, lrs, np, X_train, y_train, X_test, y_test,
                     dataset)
grid.arrange(out_mnist[[1]], out_mnist[[2]], out_mnist[[3]],
             ncol=3)
