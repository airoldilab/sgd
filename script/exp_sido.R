# Run logistic regression on SIDO0 dataset
# The dataset can be downloaded from
#   http://www.causality.inf.ethz.ch/data/SIDO.html
# To run this script, the working directory should be the base repo
#   data files should be stored in "data/"
# * sido0_train.data
# * sido0_train.targets

library(sgd)
library(gridExtra)

source("script/plot.R")
source("script/run_exp.R")

raw <- read.table("data/sido0_train.data", header=F)
labels <- read.table("data/sido0_train.targets", header=F)[, 1]
labels[labels != 1] <- 0

# Subset to work on.
set.seed(42)
idxs <- sample(1:nrow(raw), floor(0.75*nrow(raw)))
test_idxs <- 1:nrow(raw)[-idxs]

X_train <- as.matrix(raw[idxs, ])
y_train <- labels[idxs]
X_test <- as.matrix(raw[test_idxs, ])
y_test <- labels[test_idxs]

# Arguments for main function.
methods <- list("sgd", "implicit", "asgd", "ai-sgd", "sgd")
lrs <- list("one-dim", "one-dim", "one-dim", "one-dim", "adagrad")
lr.controls <- NULL
np <- list(2, 2, 2, 2, 2)
names <- list("sgd", "implicit", "asgd", "ai-sgd", "adagrad")
dataset <- "sido"
ylim <- NULL

# TODO temp, since adagrad is so slow
methods <- list("sgd", "implicit", "asgd", "ai-sgd")

out_sido <- run_exp(methods, names, lrs, lr.controls, np,
                    X_train, y_train, X_test, y_test,
                    dataset)
grid.arrange(out_sido[[1]], out_sido[[2]], out_sido[[3]],
             ncol=3)
