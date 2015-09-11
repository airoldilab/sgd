# Run logistic regression on delta dataset
# The dataset can be downloaded from
#   ftp://largescale.ml.tu-berlin.de/largescale
# To run this script, the working directory should be the base repo
#   data files should be stored in "data/"
# * delta_train.dat.bz2
# * delta_train.lab.bz2

library(sgd)
library(gridExtra)

source("demo/odyssey/plot.R")
source("demo/odyssey/run_exp.R")

# N = 500000
# raw <- read.table("data/delta_train.dat.bz2", header=F, nrows=N)
# labels <- read.table("data/delta_train.lab.bz2", header=F, nrows=N)
# labels <- as.numeric(labels[1:nrow(labels), 1])
# save(raw, labels, file="data/delta.Rdata")
load("data/delta.Rdata")

# Subset to work on.
set.seed(42)
#idxs <- sample(1:nrow(raw), floor(0.75*nrow(raw)))
#test_idxs <- 1:nrow(raw)[-idxs]
idxs <- sample(1:nrow(raw), floor(0.01*nrow(raw))) # using very small training set
test_idxs <- sample(1:nrow(raw), floor(0.01*nrow(raw))) # using small testing set

X_train <- as.matrix(raw[idxs, ])
y_train <- labels[idxs]
X_test <- as.matrix(raw[test_idxs, ])
y_test <- labels[test_idxs]

# Arguments for main function.
# lr.controls are optimized according to a grid search on a subset of the data.
methods <- list("sgd", "implicit", "asgd", "ai-sgd", "sgd")
lrs <- list("one-dim", "one-dim", "one-dim", "one-dim", "adagrad")
lr.controls <- NULL
lambda2s <- list(1e-2, 1e-2, 1e-2, 1e-2, 1e-2)
np <- list(2, 2, 2, 2, 2)
names <- list("sgd", "implicit", "asgd", "ai-sgd", "adagrad")
dataset <- "delta"
ylim <- NULL

out_delta <- run_exp(methods, names, lrs, lr.controls, np,
                     X_train, y_train, X_test, y_test,
                     dataset)
grid.arrange(out_delta[[1]], out_delta[[2]], out_delta[[3]],
             ncol=3)
