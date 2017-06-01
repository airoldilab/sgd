#!/usr/bin/env Rscript
# Compare out-of-sample log-likelihoods, using sgd() and glm() for
# logistic regression on the wine quality data set.
#
# Dimensions:
#   n=4286, d=12

library(sgd)

# Generate data.
data("winequality")
dat <- winequality
dat$quality <- as.numeric(dat$quality > 5) # transform to binary

test.set <- sample(1:nrow(dat), size=nrow(dat)/8, replace=F)
dat.test <- dat[test.set, ]
dat <- dat[-test.set, ]

# Fit glm() and sgd().
fit.glm <- glm(quality~., family=binomial(link="logit"), data=dat)
fit.sgd <- sgd(quality ~ ., data=dat,
               model="glm", model.control=binomial(link="logit"),
               sgd.control=list(reltol=1e-5, npasses=200), lr.control=c(scale=1, gamma=1, alpha=30, c=1))

# Compare log likelihoods.
log.lik <- function(theta.est) {
  
  y <- dat.test$quality
  X <- as.matrix(dat.test[, seq(1, ncol(dat)-1)])
  X <- cbind(1, X)

  eta <- plogis(X %*% theta.est)
  print(cor(y, eta))
  sum(y * log(eta) + (1-y) * log(1-eta))
}

theta.glm <- matrix(as.numeric(fit.glm$coefficients), ncol=1)
theta.sgd <- matrix(as.numeric(fit.sgd$coefficients), ncol=1)
log.lik.glm <- log.lik(fit.glm$coefficients)
log.lik.sgd <- log.lik(theta.sgd)

print(sprintf("Out-of-sample Log-likelihood for glm()=%.3f  sgd=%.3f", log.lik.glm, log.lik.sgd))
