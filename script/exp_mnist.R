# Run logistic regression on MNIST dataset
# The dataset can be downloaded from
#   http://yann.lecun.com/exdb/mnist/
# To run this script, the working directory should be the base repo
#   data files should be stored in "data/"
# * t10k-images.idx3-ubyte
# * t10k-labels.idx1-ubyte
# * train-images.idx3-ubyte
# * train-labels.idx1-ubyte

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

# Subset to work on.
set.seed(42)
idxs <- sample(1:nrow(X_train), floor(0.10*nrow(X_train))) # using small training set
test_idxs <- sample(1:nrow(X_test), floor(0.10*nrow(X_test))) # using small testing set

X_train <- X_train[idxs, ]
y_train <- y_train[idxs]
X_test <- X_test[test_idxs, ]
y_test <- y_test[test_idxs]

# Set task to be binary classification on digit 9.
y_train[y_train != 9] <- 0
y_train[y_train == 9] <- 1
y_test[y_test != 9] <- 0
y_test[y_test == 9] <- 1

# Arguments for main function.
methods <- list("sgd", "implicit", "asgd", "ai-sgd", "sgd")
lrs <- list("one-dim", "one-dim", "one-dim", "one-dim", "adagrad")
lr.controls <- list(0.025, 0.025, 0.025, 0.025, NULL)
lambda2s <- list(1e-3, 1e-3, 1e-3, 1e-3, 1e-3)
np <- list(2, 2, 2, 2, 2)
names <- list("sgd", "implicit", "asgd", "ai-sgd", "adagrad")
dataset <- "mnist"
ylim <- list(c(0.025, 0.075), c(0.025, 0.075), c(0,2))

out_mnist <- run_exp(methods, names, lrs, lr.controls, np,
                     X_train, y_train, X_test, y_test,
                     dataset, ylim)
grid.arrange(out_mnist[[1]], out_mnist[[2]], out_mnist[[3]],
             ncol=3)
