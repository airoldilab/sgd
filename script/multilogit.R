library(sgd)
library(ggplot2)


multilogit.fit <- function(X, y, ...){
  
  labels <- unique(y)
  nlabels <- length(unique(y))
  pivot.label <- labels[1]
  coefs <- array(0, dim=c(nlabels, ncol(X)+1, 100))
  pos <- array(0, dim=c(nlabels-1, 100))
  count <- 2
  for (label in labels[2:nlabels]){
    X_temp <- X[y==label | y==pivot.label, ]
    y_temp <- y[y==label | y==pivot.label]
    select <- y_temp==label
    y_temp[!select] <- 0
    y_temp[select] <- 1
    X_temp <- cbind(rep(1, nrow(X_temp)), X_temp)
    model <- sgd(X_temp, y_temp, "glm", model.control=list(family="binomial"), ...)
    coefs[count, , ] <- model$estimates
    pos[count-1, ] <- model$pos
    count <- count + 1
  }
  return(list(coefs=coefs, pos=pos, labels=labels))
}

multilogit.predict <- function(model, X){
  X <- cbind(rep(1, nrow(X)), X)
  etas <- array(0, dim=c(dim(model$coefs)[1], nrow(X), dim(model$coefs)[3]))
  # TODO: vectorize this
  for (i in 1:dim(model$coefs)[1]){
    etas[i, , ] <- exp(X %*% model$coefs[i, , ])
  }
  # TODO: vectorize this
  pred <- apply(etas, c(2,3), function(x) model$labels[which.max(x)])
  prob <- apply(etas, c(2,3), function(x) x/sum(x))
  return(list(pred=pred, pos=model$pos, prob=prob))
}
