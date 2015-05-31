# Run logistic regression on Covertype dataset
# The dataset can be downloaded from
#   https://archive.ics.uci.edu/ml/datasets/Covertype
# To run this script, the working directory should be the base repo
#   data files should be stored in "data/"

source("script/plot.R")
source("script/run_exp.R")

library(sgd)
library(gridExtra)

set.seed(42)
raw <- read.table("data/covtype.data", sep=",")
idxs <- sample(1:nrow(raw), floor(0.80*nrow(raw)))
raw_train <- raw[idxs, ]
raw_test <- raw[-idxs, ]

X_train <- as.matrix(raw_train[, -55])
y_train <- raw_train[, 55]
X_test <- as.matrix(raw_test[, -55])
y_test <- raw_test[, 55]

# Set task to be binary classification on class 2.
y_train[y_train != 2] <- 0
y_train[y_train == 2] <- 1
y_test[y_test != 2] <- 0
y_test[y_test == 2] <- 1

methods <- list("implicit")
lrs <- list("one-dim")
np <- list(1)
names <- methods
dataset <- "covtype"

out_covtype <- run_exp(methods, names, lrs, np, X_train, y_train, X_test,
                       y_test, dataset)
grid.arrange(out_covtype[[1]], out_covtype[[2]], out_covtype[[3]],
             ncol=3)
