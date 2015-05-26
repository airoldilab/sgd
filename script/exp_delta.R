


library(sgd)
# N = 500000
# raw <- read.table("data/delta_train.dat.bz2", header = F, nrows=N)
# labels <- read.table("data/delta_train.lab.bz2", header = F, nrows=N)
# labels <- as.numeric(labels[1:nrow(labels), 1])
# save(raw, labels, file="data/delta.Rdata")

load("data/delta.Rdata")

idxs <- sample(1:nrow(raw), floor(0.75*nrow(raw)))
# idxs <- sample(1:nrow(raw), floor(0.01*nrow(raw))) # using very small training set

X <- as.matrix(raw[idxs, ])
y <- labels[idxs]
X_test <- as.matrix(raw[-idxs, ])
y_test <- labels[-idxs]

methods <- list("implicit")
lrs <- list("one-dim")
np <- list(10)
names <- methods
run_exp(methods, names, lrs, np, X, y, X_test, y_test)
