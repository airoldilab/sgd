# Run logistic regression on MNIST dataset
# The dataset can be downloaded from
#   http://yann.lecun.com/exdb/mnist/
# To run this script, the working directory should be the base repo
#   data files should be stored in "data/"
# * t10k-images.idx3-ubyte
# * t10k-labels.idx1-ubyte
# * train-images.idx3-ubyte
# * train-labels.idx1-ubyte

source("demo/odyssey/section_3.2_mnist_load.R")

dat <- load_mnist()
X_train <- dat$train$x
y_train <- dat$train$y
X_test <- dat$test$x
y_test <- dat$test$y

# Set task to be binary classification on digit 9.
y_train[y_train != 9] <- 0
y_train[y_train == 9] <- 1
y_test[y_test != 9] <- 0
y_test[y_test == 9] <- 1

# Set up generic X, y.
X_train[1,1] <- as.numeric(X_train[1,1])
y_train <- as.numeric(y_train)
X <- X_train
y <- y_train
