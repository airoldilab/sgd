source("script/plot.R")

library(sgd)
library(ggplot2)

multilogit.fit <- function(X, y, ...) {
  labels <- unique(y)
  nlabels <- length(unique(y))
  pivot.label <- labels[1]
  coefs <- array(0, dim=c(nlabels, ncol(X)+1, 100))
  pos <- array(0, dim=c(nlabels-1, 100))
  times <- matrix(0, nrow=nlabels-1, ncol=100)
  for (count in 2:nlabels) {
    label <- labels[count]
    X_temp <- X[y==label | y==pivot.label, ]
    y_temp <- y[y==label | y==pivot.label]
    select <- y_temp==label
    y_temp[!select] <- 0
    y_temp[select] <- 1
    X_temp <- cbind(rep(1, nrow(X_temp)), X_temp)
    model <- sgd(X_temp, y_temp, "glm",
                 model.control=list(family="binomial"),
                 ...)
    print(sprintf("Finish fitting %d out of %d labels; Time: %f s",
          count-1, nlabels-1, model$times[length(model$times)]))
    coefs[count, , ] <- model$estimates
    pos[count-1, ] <- model$pos
    times[count-1, ] <- model$times
  }
  times <- colMeans(times) # output average time for each iteration t, ranging
                           # over each sgd fit of binary classification
  return(list(coefs=coefs, pos=pos, labels=labels, times=times))
}

multilogit.predict <- function(model, X) {
  X <- cbind(rep(1, nrow(X)), X)
  etas <- array(0, dim=c(dim(model$coefs)[1], nrow(X), dim(model$coefs)[3]))
  # TODO: vectorize this
  for (i in 1:dim(model$coefs)[1]){
    etas[i, , ] <- exp(X %*% model$coefs[i, , ])
  }
  # TODO: vectorize this
  pred <- apply(etas, c(2,3), function(x) model$labels[which.max(x)])
  prob <- apply(etas, c(2,3), function(x) x/sum(x))
  return(list(pred=pred, pos=model$pos, prob=prob, labels=model$labels))
}

run_exp <- function(methods, names, lrs, np, X, y, X_test=X, y_test=y, plot=T) {

  # Args:
  #  methods: a list of "sgd", "implicit" or "ai-sgd"
  #  names: a list of labels for each experiment for plotting
  #  lrs: a list of learning rate types
  #  np: a list of number of passes

  models <- list()
  preds <- list()
  pred_trains <- list()
  y_tests <- list()
  y_trains <- list()
  times <- list()
  for (i in 1:length(methods)) {
    time_start <- proc.time()
    model <- multilogit.fit(X, y, sgd.control=list(
      method=methods[[i]], lr=lrs[[i]], npasses=np[[i]]))
    times[[i]] <- model$times
    pred <- multilogit.predict(model, X_test)
    pred_train <- multilogit.predict(model, X)
    models[[i]] <- model
    preds[[i]] <- pred
    pred_trains[[i]] <- pred_train
    y_tests[[i]] <- y_test
    y_trains[[i]] <- y
    time <- proc.time()[3] - time_start
    print(sprintf("experiment %d of %d done! Time: %f s", i, length(methods), time))
  }
  if (plot) {
    return(list(
      plot.error(preds, y_tests, names),
      plot.cost(pred_trains, y_trains, names),
      plot.error.runtime(preds, y_tests, names, times)))
  } else {
    return(list(models=models, preds=preds))
  }
}
