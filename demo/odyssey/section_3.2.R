# SECTION 3.2, TABLE 3
# Choose data source here.
#source("demo/odyssey/section_3.2_mnist.R")
#source("demo/odyssey/section_3.2_covtype.R")
#source("demo/odyssey/section_3.2_delta.R")

# sgd
library(sgd)
time_start <- proc.time()[3]
theta.sgd <- sgd(X, y, model="glm", model.control=list(family="binomial"),
  sgd.control=list(npass=1, pass=T, size=1))
time.sgd <- as.numeric(proc.time()[3] - time_start)

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

# speedglm
library(speedglm)
#theta.speedglm <- speedglm.wfit(y=y, X=X, family=binomial(link=logit), chunk=500)
time_start <- proc.time()[3]
theta.speedglm <- speedglm.wfit(y=y, X=X, family=binomial(link=logit))
time.speedglm <- as.numeric(proc.time()[3] - time_start)

# liblinear
library(LiblineaR)
time_start <- proc.time()[3]
theta.liblinear <- LiblineaR(data=X, target=y, type=7)
time.liblinear <- as.numeric(proc.time()[3] - time_start)

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

# glm.fit
time_start <- proc.time()[3]
theta.glm <- glm.fit(x=X, y=y, family=binomial(link=logit))
time.glm <- as.numeric(proc.time()[3] - time_start)
