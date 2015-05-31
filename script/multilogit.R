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
    print(sprintf("Finish training %d out of %d labels; Time: %f s",
          count-1, nlabels-1, model$times[length(model$times)]))
    coefs[count, , ] <- model$estimates
    pos[count-1, ] <- model$pos
    times[count-1, ] <- model$times
  }
  times <- colMeans(times)
  return(list(coefs=coefs, pos=pos, labels=labels, times=times))
}

multilogit.predict <- function(model, X) {
  time_start <- proc.time()[3]
  X <- cbind(rep(1, nrow(X)), X)
  etas <- array(0, dim=c(dim(model$coefs)[1], nrow(X), dim(model$coefs)[3]))
  for (i in 1:dim(model$coefs)[1]) {
    etas[i, , ] <- exp(X %*% model$coefs[i, , ])
  }
  # TODO
  #prob <- array(NA, dim=dim(etas))
  #pred <- matrix(NA, nrow=dim(etas)[2], ncol=dim(etas)[3])
  #for (k in 1:(dim(etas)[3])) {
  #  #prob[, , k] <- etas[, , k]/colSums(etas[, , k])
  #  for (j in 1:(dim(etas)[2])) {
  #    prob[, j, k] <- etas[, j, k]/sum(etas[, j, k])
  #    pred[j, k] <- model$labels[which.max(etas[, j, k])]
  #  }
  #}
  pred <- apply(etas, c(2,3), function(x) model$labels[which.max(x)])
  prob <- apply(etas, c(2,3), function(x) x/sum(x))
  prob[prob < 1e-8] <- 0
  prob[is.nan(prob)] <- 1
  print(sprintf("Finish testing; Time: %f s",
        proc.time()[3] - time_start))
  return(list(pred=pred, pos=model$pos, prob=prob, labels=model$labels))
}

run_exp <- function(methods, names, lrs, np, X_train, y_train, X_test=NULL,
                    y_test=NULL, plot=T) {

  # Args:
  #  methods: a list of sgd methods
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
    time_start <- proc.time()[3]
    model <- multilogit.fit(X_train, y_train, sgd.control=list(
      method=methods[[i]], lr=lrs[[i]], npasses=np[[i]]))
    models[[i]] <- model
    times[[i]] <- model$times
    pred_train <- multilogit.predict(model, X_train)
    pred_trains[[i]] <- pred_train
    y_trains[[i]] <- y_train
    if (!is.null(X_test) && !is.null(y_test)) {
      pred <- multilogit.predict(model, X_test)
      preds[[i]] <- pred
      y_tests[[i]] <- y_test
    } else {
      preds[[i]] <- pred_train
      y_tests[[i]] <- y_train
    }
    time <- proc.time()[3] - time_start
    print(sprintf("Experiment %d (%s) of %d done! Time: %f s",
                  i, methods[i], length(methods), time))
  }
  if (plot) {
    return(list(
      plot.error(preds, y_tests, names, np, title="Test error"),
      plot.cost(pred_trains, y_trains, names, np, title="Training cost"),
      plot.error.runtime(preds, y_tests, names, times, title="Test error")))
  } else {
    return(list(models=models, preds=preds))
  }
}
