# Run logistic regression on Covertype dataset
# The dataset can be downloaded from
#   https://archive.ics.uci.edu/ml/datasets/Covertype
# To run this script, the working directory should be the base repo
#   data files should be stored in "data/"

source("script/plot.R")
source("script/run_exp.R")

library(sgd)
library(gridExtra)

raw <- read.table("data/covtype.data", sep=",")

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

methods <- list("sgd", "implicit", "sgd", "ai-sgd")
lrs <- list("one-dim", "one-dim", "adagrad", "adagrad")
lr.controls <- list(0.00025, 0.00025, NULL, 0.00025)
np <- list(5, 5, 5, 5)
names <- list("sgd", "implicit", "adagrad", "ai-sgd")
dataset <- "covtype"
ylim <- list(c(0.25, 0.45), c(0.25, 0.45), NULL)

out_covtype <- run_exp(methods, names, lrs, lr.controls, np,
                       X_train, y_train, X_test, y_test,
                       dataset, ylim)
grid.arrange(out_covtype[[1]], out_covtype[[2]], out_covtype[[3]],
             ncol=3)
