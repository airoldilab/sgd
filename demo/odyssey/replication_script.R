#!/usr/bin/env Rscript
################################################################################
# SECTION 3.1, TABLE 1
################################################################################
# Benchmark sgd package for linear regression on simulated data from a
# correlated normal distribution. This follows the experiment in Section 5.1 of
# Friedman et al. (2010).
#
# Data generating process:
#   Y = sum_{j=1}^p X_j*β_j + k*ɛ, where
#     X ~ Multivariate normal where each covariate Xj, Xj' has equal correlation
#       ρ; ρ ranges over (0,0.1,0.2,0.5,0.9,0.95) for each pair (n, p)
#     β_j = (-1)^j exp(-2(j-1)/20)
#     ɛ ~ Normal(0,1)
#     k = 3
#
# Dimensions:
#   n=1,000, d=100
#   n=10,000, d=1,000
#   n=50,000, d=10,000
#   n=1,000,000, d=50,000
#   n=10,000,000, d=100,000

library(sgd)
library(glmnet)

# Function taken from Friedman et al.
genx = function(n,p,rho){
  #    generate x's multivariate normal with equal corr rho
  # Xi = b Z + Wi, and Z, Wi are independent normal.
  # Then Var(Xi) = b^2 + 1
  #  Cov(Xi, Xj) = b^2  and so cor(Xi, Xj) = b^2 / (1+b^2) = rho
  z=rnorm(n)
  if(abs(rho)<1){
    beta=sqrt(rho/(1-rho))
    x=matrix(rnorm(n*p),ncol=p)
    A = matrix(rnorm(n), nrow=n, ncol=p, byrow=F)
    x= beta * A + x
  }
  if(abs(rho)==1){ x=matrix(rnorm(n),nrow=n,ncol=p,byrow=F)}

  return(x)
}

# Dimensions: Put them manually here!
nSim <- 1    # number of runs
N <- 5e4      # size of minibatch
nstreams <- 1 # number of streams
d <- 1e4
rho <- 0

times.aisgd <- rep(0, nSim)
times.sgd <- rep(0, nSim)
times.glmnet <- rep(0, nSim)
converged.aisgd <- FALSE
converged.sgd <- FALSE

set.seed(42)
for (i in 1:nSim) {
  print(sprintf("Running simulation %i of %i...", i, nSim))
  for (nstream in 1:nstreams) {
    # Generate stream of data.
    X <- genx(N, d, rho)
    theta <- ((-1)^(1:d))*exp(-2*((1:d)-1)/20)
    eps <- rnorm(N)
    k <- 3
    y <- X %*% theta + k * eps

    # AI-SGD
    if (!converged.aisgd) {
      if (nstream == 1) {
        start <- rnorm(d, mean=0, sd=1e-5)
      } else {
        start <- aisgd.theta$coefficients
      }
      aisgd.theta <- sgd(X, y, model="lm",
        sgd.control=list(method="implicit", npasses=1, pass=T, start=start))
      times.aisgd[i] <- times.aisgd[i] + max(aisgd.theta$times)

      converged.aisgd <- aisgd.theta$converged
      if (converged.aisgd) {
        print(sprintf("AI-SGD converged early! On nstream %i", nstream))
      }
    }

    # explicit SGD
    if (!converged.sgd) {
      if (nstream == 1) {
        start <- start # using same as AI-SGD's start
      } else {
        start <- sgd.theta$coefficients
      }
      sgd.theta <- sgd(X, y, model="lm",
        sgd.control=list(method="sgd", npasses=1, pass=T, lr.control=c(0.1,
        NA, NA, NA), start=start))
      times.sgd[i] <- times.sgd[i] + max(sgd.theta$times)

      converged.sgd <- sgd.theta$converged
      if (converged.sgd) {
        print(sprintf("SGD converged early! On nstream %i", nstream))
      }
    }

    # glmnet doesn't work on streaming data
    if (nstreams == 1) {
      time_start <- proc.time()[3]
      glmnet.theta <- glmnet(X, y, alpha=1, standardize=FALSE,
        type.gaussian="covariance")
      times.glmnet[i] <- as.numeric(proc.time()[3] - time_start)
    }
  }
}
print(mean(times.aisgd * 1e1)) # convert nlambdas
print(mean(times.sgd * 1e1))
print(mean(times.glmnet))

################################################################################
# SECTION 3.2, TABLE 3
################################################################################

run.fit <- function(X, y) {
  # sgd
  library(sgd)
  time_start <- proc.time()[3]
  theta.sgd <- sgd(X, y, model="glm", model.control=list(family="binomial"),
    sgd.control=list(npass=1, pass=T, size=1))
  time.sgd <- as.numeric(proc.time()[3] - time_start)
  print(sprintf("Time (s) for sgd: %0.3f", time.sgd))

  # biglm
  library(biglm)
  colnames(X) <- NULL
  dat <- data.frame(y=y, X=X)
  xvarname <- paste("X.", 1:(ncol(dat)-1), sep="")
  ff <- as.formula(paste("y ~ ", paste(xvarname, collapse="+")))
  time_start <- proc.time()[3]
  theta.biglm <- bigglm(ff, data=dat, family=binomial(link="logit"), maxit=20,
  chunksize=10000)
  time.biglm <- as.numeric(proc.time()[3] - time_start)
  print(sprintf("Time (s) for biglm: %0.3f", time.biglm))

  # speedglm
  library(speedglm)
  #theta.speedglm <- speedglm.wfit(y=y, X=X, family=binomial(link=logit), chunk=500)
  time_start <- proc.time()[3]
  theta.speedglm <- speedglm.wfit(y=y, X=X, family=binomial(link=logit))
  time.speedglm <- as.numeric(proc.time()[3] - time_start)
  print(sprintf("Time (s) for speedglm: %0.3f", time.speedglm))

  # liblinear
  library(LiblineaR)
  time_start <- proc.time()[3]
  theta.liblinear <- LiblineaR(data=X, target=y, type=7)
  time.liblinear <- as.numeric(proc.time()[3] - time_start)
  print(sprintf("Time (s) for liblinear: %0.3f", time.liblinear))

  # mnlogit
  library(mnlogit)
  library(mlogit)
  colnames(X) <- NULL
  dat <- data.frame(y=y, X=X)
  row.names(dat) <- NULL
  dat <- mlogit.data(dat, choice="y", shape="wide")
  xvarname <- paste("X.", 1:(ncol(dat)-3), sep="")
  ff <- as.formula(paste("y ~ ", paste(xvarname, collapse="+")))
  time_start <- proc.time()[3]
  theta.mnlogit <- mnlogit(ff, dat)
  time.mnlogit <- as.numeric(proc.time()[3] - time_start)
  print(sprintf("Time (s) for mnlogit: %0.3f", time.mnlogit))

  # glm.fit
  time_start <- proc.time()[3]
  theta.glm <- glm.fit(x=X, y=y, family=binomial(link=logit))
  time.glm <- as.numeric(proc.time()[3] - time_start)
  print(sprintf("Time (s) for glm: %0.3f", time.glm))
}

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

# Run all methods.
run.fit(X, y)

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
y_train[y_train == -1] <- 0
X <- X_train
y <- y_train

# Run all methods.
run.fit(X, y)

# Run logistic regression on MNIST dataset
# The dataset can be downloaded from
#   http://yann.lecun.com/exdb/mnist/
# To run this script, the working directory should be the base repo
#   data files should be stored in "data/"
# * t10k-images.idx3-ubyte
# * t10k-labels.idx1-ubyte
# * train-images.idx3-ubyte
# * train-labels.idx1-ubyte

load_image_file <- function(filename) {
  ret = list()
  f = file(filename,'rb')
  readBin(f,'integer',n=1,size=4,endian='big')
  ret$n = readBin(f,'integer',n=1,size=4,endian='big')
  nrow = readBin(f,'integer',n=1,size=4,endian='big')
  ncol = readBin(f,'integer',n=1,size=4,endian='big')
  x = readBin(f,'integer',n=ret$n*nrow*ncol,size=1,signed=F)
  ret$x = matrix(x, ncol=nrow*ncol, byrow=T)
  close(f)
  ret
}
load_label_file <- function(filename) {
  f = file(filename,'rb')
  readBin(f,'integer',n=1,size=4,endian='big')
  n = readBin(f,'integer',n=1,size=4,endian='big')
  y = readBin(f,'integer',n=n,size=1,signed=F)
  close(f)
  y
}
show_digit <- function(arr784, col=gray(12:1/12), ...) {
  image(matrix(arr784, nrow=28)[,28:1], col=col, ...)
}
load_mnist <- function() {
  train <- load_image_file('demo/odyssey/data/train-images.idx3-ubyte')
  test <- load_image_file('demo/odyssey/data/t10k-images.idx3-ubyte')

  train$y <- load_label_file('demo/odyssey/data/train-labels.idx1-ubyte')
  test$y <- load_label_file('demo/odyssey/data/t10k-labels.idx1-ubyte')

  return(list(train=train, test=test))
}

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

# Run all methods.
run.fit(X, y)

################################################################################
# SECTION 3.3, TABLE 4
################################################################################
# This is used to generate the table in M-estimation experiments section.

library(sgd)
library(ggplot2)

generate.data <- function(N, d) {
  l2 <- function(x) sqrt(sum(x**2))
  X <- matrix(rnorm(N*d, mean=0, sd=1/sqrt(N)), nrow=N, ncol=d)
  theta <- runif(d)
  theta <- theta * 6 *sqrt(d) / l2(theta)

  # noise
  ind <- rbinom(N, size=1, prob=.95)
  epsilon <- ind * rnorm(N) + (1-ind) * rep(10 ,N)

  Y <- X %*% theta + epsilon
  return(list(y=Y, X=X, theta=theta))
}

# Dimensions
N <- 1e4
d <- 5e2

# Generate data.
set.seed(42)
data <- generate.data(N, d)
dat <- data.frame(y=data$y, x=data$X)

times.sgd <- c()
times.aisgd <- c()
times.hqreg <- c()

nSim <- 10
for (i in 1:nSim) {
  sgd.theta1 <- sgd(y ~ .-1, data=dat, model="m",
    sgd.control=list(
    method="sgd",
    lr.control=c(15, NA, NA, 1/2), npass=1, pass=T, size=1, start=rep(5,d)))
  times.sgd <- c(times.sgd, sgd.theta1$times)
  sgd.theta2 <- sgd(y ~ .-1, data=dat, model="m",
    sgd.control=list(
    method="ai-sgd",
    lr.control=c(15, NA, NA, 2/3), npass=1, pass=T, size=1, start=rep(5,d)))
  times.aisgd <- c(times.aisgd, sgd.theta2$times)

  library(hqreg)
  time_start <- proc.time()[3]
  hqreg <- hqreg(data$X, as.vector(data$y), method = "huber",
    gamma=3, alpha=1)
  times.hqreg <- c(times.hqreg, as.numeric(proc.time()[3] - time_start))
}
print(mean(times.sgd))
print(mean(times.aisgd))
print(mean(times.hqreg))

################################################################################
# SECTION 3.3, FIGURE 2
################################################################################
# This is used to generate the plot in M-estimation experiments section.

library(sgd)
library(ggplot2)

generate.data <- function(N, d) {
  l2 <- function(x) sqrt(sum(x**2))
  X <- matrix(rnorm(N*d, mean=0, sd=1/sqrt(N)), nrow=N, ncol=d)
  theta <- runif(d)
  theta <- theta * 6 *sqrt(d) / l2(theta)

  # noise
  ind <- rbinom(N, size=1, prob=.95)
  epsilon <- ind * rnorm(N) + (1-ind) * rep(10 ,N)

  Y <- X %*% theta + epsilon
  return(list(y=Y, X=X, theta=theta))
}

# Dimensions
N <- 100000
d <- 10000

# Generate data.
set.seed(42)
data <- generate.data(N, d)
dat <- data.frame(y=data$y, x=data$X)

sgd.theta1 <- sgd(y ~ .-1, data=dat, model="m",
  sgd.control=list(
  method="sgd",
  lr.control=c(15, NA, NA, 1/2), npass=30, pass=T, size=0.5*N, start=rep(5,d)))
sgd.theta2 <- sgd(y ~ .-1, data=dat, model="m",
  sgd.control=list(
  method="ai-sgd",
  lr.control=c(15, NA, NA, 2/3), npass=30, pass=T, size=0.5*N, start=rep(5,d)))
sgd.theta3 <- sgd(y ~ .-1, data=dat, model="m",
  sgd.control=list(
  method="ai-sgd",
  lr="rmsprop",
  lr.control=c(0.01, NA, NA), npass=30, pass=T, size=0.5*N, start=rep(5,d)))
sgd.theta4 <- sgd(y ~ .-1, data=dat, model="m",
  sgd.control=list(
  method="sgd",
  lr="rmsprop",
  lr.control=c(0.01, NA, NA), npass=30, pass=T, size=0.5*N, start=rep(5,d)))

sgd.list <- list("ai-sgd"=sgd.theta1, "sgd"=sgd.theta2, "sgd+rmsprop"=sgd.theta4,"ai-sgd+rmsprop"=sgd.theta3)

p1 <- plot(sgd.list, data$theta, type="mse-param") +
  geom_hline(yintercept=1.5, color="green") +
  ggplot2::scale_y_continuous(
    breaks=seq(1, 10, 1),
    limits=c(1, 10))
      #legend.position=c(0.55, 0.6),
p2 <- plot(sgd.list, data$theta, type="mse-param", xaxis="runtime") +
  geom_hline(yintercept=1.5, color="green") +
  ggplot2::scale_y_continuous(
    breaks=seq(1, 10, 1),
    limits=c(1, 10))
pdf("temp/huber_1.pdf", width=4, height=4)
print(p1)
dev.off()
pdf("temp/huber_2.pdf", width=4, height=4)
print(p2)
dev.off()
