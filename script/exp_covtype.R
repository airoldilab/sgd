# Run logistic regression on  CovType dataset
# The dataset can be downloaded from
#   https://archive.ics.uci.edu/ml/datasets/Covertype
# To run this script, the working directory should be the base repo
#   data files should be stored in "data/"

library(sgd)
source("multilogit.R")
source("plot.R")


set.seed(42)
raw <- read.table("data/covtype.data", sep=",")
idxs <- sample(1:nrow(raw), floor(0.75*nrow(raw)))
# idxs <- sample(1:nrow(raw), floor(0.01*nrow(raw))) # using very small training set
raw.train <- raw[idxs, ]
raw.test <- raw[-idxs, ]

X <- as.matrix(raw.train[, -55])
y <- raw.train[, 55]
X_test <- as.matrix(raw.test[, -55])
y_test <- raw.test[, 55]

methods <- list("implicit")
lrs <- list("one-dim")
np <- list(10)
names <- methods
run_exp(methods, names, lrs, np, X, y, X_test, y_test)
