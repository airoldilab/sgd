# Run logistic regression on Covertype dataset
# The dataset can be downloaded from
#   https://archive.ics.uci.edu/ml/datasets/Covertype
# To run this script, the working directory should be the base repo
#   data files should be stored in "data/"
# * covtype.data

raw <- read.table("demo/odyssey/data/covtype.data", sep=",")

# Subset to work on.
set.seed(42)
idxs <- sample(1:nrow(raw), floor(0.80*nrow(raw)))
test_idxs <- 1:nrow(raw)[-idxs]
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

# Set up generic X, y.
X_train[1,1] <- as.numeric(X_train[1,1])
y_train <- as.numeric(y_train)
X <- X_train
y <- y_train
