# Run logistic regression on delta dataset
# The dataset can be downloaded from
#   ftp://largescale.ml.tu-berlin.de/largescale
# To run this script, the working directory should be the base repo
#   data files should be stored in "data/"
# * delta_train.dat.bz2
# * delta_train.lab.bz2

# N = 500000
# raw <- read.table("data/delta_train.dat.bz2", header=F, nrows=N)
# labels <- read.table("data/delta_train.lab.bz2", header=F, nrows=N)
# labels <- as.numeric(labels[1:nrow(labels), 1])
# save(raw, labels, file="data/delta.Rdata")
load("demo/odyssey/data/delta.Rdata")

# Subset to work on.
set.seed(42)
idxs <- sample(1:nrow(raw), floor(0.75*nrow(raw)))
test_idxs <- 1:nrow(raw)[-idxs]

X_train <- as.matrix(raw[idxs, ])
y_train <- labels[idxs]
X_test <- as.matrix(raw[test_idxs, ])
y_test <- labels[test_idxs]

# Set up generic X, y.
X_train[1,1] <- as.numeric(X_train[1,1])
y_train <- as.numeric(y_train)
X <- X_train
y <- y_train
