# Run logistic regression on Covertype dataset
# The dataset can be downloaded from
#   https://archive.ics.uci.edu/ml/datasets/Covertype
# To run this script, the working directory should be the base repo
#   data files should be stored in "data/"
# * covtype.data

source("script/plot.R")
source("script/run_exp.R")

library(sgd)
library(gridExtra)

raw <- read.table("data/covtype.data", sep=",")

# Subset to work on.
set.seed(42)
#idxs <- sample(1:nrow(raw), floor(0.80*nrow(raw)))
#test_idxs <- 1:nrow(raw)[-idxs]
idxs <- sample(1:nrow(raw), floor(0.09*nrow(raw))) # using small training set
test_idxs <- sample(1:nrow(raw), floor(0.01*nrow(raw))) # using small testing set
raw_train <- raw[idxs, ]
raw_test <- raw[test_idxs, ]

X_train <- as.matrix(raw_train[, -55])
y_train <- raw_train[, 55]
X_test <- as.matrix(raw_test[, -55])
y_test <- raw_test[, 55]

# Set task to be binary classification on class 2.
y_train[y_train != 2] <- 0
y_train[y_train == 2] <- 1
y_test[y_test != 2] <- 0
y_test[y_test == 2] <- 1

# Arguments for main function.
methods <- list("sgd", "implicit", "asgd", "ai-sgd", "sgd")
lrs <- list("one-dim", "one-dim", "one-dim", "one-dim", "adagrad")
lr.controls <- list(0.00025, 0.00025, 0.00025, 0.00025, NULL)
lambda2s <- list(1e-6, 1e-6, 1e-6, 1e-6, 1e-6)
np <- list(5, 5, 5, 5, 5)
names <- list("sgd", "implicit", "asgd", "ai-sgd", "adagrad")
dataset <- "covtype"
ylim <- list(c(0.25, 0.45), c(0.25, 0.45), NULL)

out_covtype <- run_exp(methods, names, lrs, lr.controls, lambda2s, np,
                       X_train, y_train, X_test, y_test,
                       dataset, ylim)
grid.arrange(out_covtype[[1]], out_covtype[[2]], out_covtype[[3]],
             ncol=3)
