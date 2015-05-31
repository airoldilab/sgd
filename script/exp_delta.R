# Run logistic regression on delta dataset
# The dataset can be downloaded from
#   ftp://largescale.ml.tu-berlin.de/largescale
# To run this script, the working directory should be the base repo
#   data files should be stored in "data/"

library(sgd)
library(gridExtra)

source("script/plot.R")
source("script/run_exp.R")

# N = 500000
# raw <- read.table("data/delta_train.dat.bz2", header=F, nrows=N)
# labels <- read.table("data/delta_train.lab.bz2", header=F, nrows=N)
# labels <- as.numeric(labels[1:nrow(labels), 1])
# save(raw, labels, file="data/delta.Rdata")
load("data/delta.Rdata")

#idxs <- sample(1:nrow(raw), floor(0.75*nrow(raw)))
#test_idxs <- 1:nrow(raw)[-idxs]
idxs <- sample(1:nrow(raw), floor(0.01*nrow(raw))) # using very small training set
test_idxs <- sample(1:nrow(raw), floor(0.01*nrow(raw))) # using small testing set

X_train <- as.matrix(raw[idxs, ])
y_train <- labels[idxs]
X_test <- as.matrix(raw[test_idxs, ])
y_test <- labels[test_idxs]

methods <- list("implicit")
lrs <- list("one-dim")
np <- list(10)
names <- methods
dataset <- "delta"

out_delta <- run_exp(methods, names, lrs, np, X_train, y_train, X_test, y_test,
                     dataset)
grid.arrange(out_delta[[1]], out_delta[[2]], out_delta[[3]],
             ncol=3)
