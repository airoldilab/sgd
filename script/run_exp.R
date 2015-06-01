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
  if (dim(model$coefs)[1] == 2) { # make computation easier in binary case
    prob <- array(NA, dim=c(dim(model$coefs)[1], nrow(X), dim(model$coefs)[3]))
    prob[2, , ] <- 1 / (1 + exp(-X %*% model$coefs[2, , ]))
    prob[prob < 1e-16] <- 1e-16
    prob[prob > 1-1e-16] <- 1-1e-16
    prob[1, , ] <- 1 - prob[2, , ]
  } else {
    # TODO numerical stability issues due to exp(X*B)/(1 + sum(exp(X*B)))
    etas <- array(0, dim=c(dim(model$coefs)[1], nrow(X), dim(model$coefs)[3]))
    for (i in 1:dim(model$coefs)[1]) {
      if (i == 1) {
        etas[i, , ] <- 1
      } else {
        etas[i, , ] <- exp(X %*% model$coefs[i, , ])
      }
    }
    # Calculate probability, and truncate if < eps, > 1-eps, or NaN.
    prob <- array(NA, dim=dim(etas))
    for (k in 1:(dim(etas)[3])) {
      prob[, , k] <- etas[, , k]/colSums(etas[, , k])
    }
    #prob[prob < 1e-8] <- 1e-8
    #prob[prob > 1-1e-8] <- 1 - 1e-8
    prob[is.nan(prob)] <- 1-1e-5
  }
  # Predict label based on highest probability.
  pred <- matrix(NA, nrow=dim(prob)[2], ncol=dim(prob)[3])
  for (k in 1:(dim(prob)[3])) {
    for (j in 1:(dim(prob)[2])) {
      # note: does not break ties randomly; it chooses the first index
      pred[j, k] <- model$labels[which.max(prob[, j, k])]
    }
  }
  if (dim(model$coefs)[1] != 2) {
    prob[prob < 1e-8] <- 1e-8
    #prob[prob > 1-1e-8] <- 1-1e-8
  }
  print(sprintf("Finish testing; Time: %f s",
        proc.time()[3] - time_start))
  return(list(pred=pred, pos=model$pos, prob=prob, labels=model$labels))
}

run_exp <- function(methods, names, lrs, lr.controls=NULL, lambda2s=NULL, np,
                    X_train, y_train, X_test=NULL, y_test=NULL,
                    dataset=NULL, ylim=NULL, plot=T) {

  # Args:
  #  methods: a list of sgd methods
  #  names: a list of labels for each experiment for plotting
  #  lrs: a list of learning rate types
  #  np: a list of number of passes
  #  dataset: character string specifying name of experiment
  #  ylim: list of 3 two-vectors which y-limits for each plot

  models <- list()
  preds <- list()
  y_tests <- list()
  times <- list()
  for (i in 1:length(methods)) {
    time_start <- proc.time()[3]
    model <- multilogit.fit(X_train, y_train,
      sgd.control=list(method=methods[[i]], lr=lrs[[i]],
                       lr.control=lr.controls[[i]], npasses=np[[i]],
                       lambda2=lambda2s[[i]]))
    models[[i]] <- model
    times[[i]] <- model$times
    if (!is.null(X_test) && !is.null(y_test)) {
      pred <- multilogit.predict(model, X_test)
      preds[[i]] <- pred
      y_tests[[i]] <- y_test
    } else {
      pred_train <- multilogit.predict(model, X_train)
      preds[[i]] <- pred_train
      y_tests[[i]] <- y_train
    }
    time <- proc.time()[3] - time_start
    print(sprintf("Experiment %d (%s) of %d done! Time: %f s",
                  i, names[i], length(methods), time))
  }
  if (plot) {
    if (is.null(dataset)) {
      titles <- c("Test error", "Test error", "Test cost")
    } else {
      titles <- c(sprintf("%s test error", dataset),
                  sprintf("%s test error", dataset),
                  sprintf("%s test cost", dataset))
    }
    return(list(
      plot.error(preds, y_tests, names, np, title=titles[1], ylim=ylim[[1]]),
      plot.error.runtime(preds, y_tests, names, times, title=titles[2],
                         ylim=ylim[[2]]),
      plot.cost(preds, y_tests, names, np, title=titles[3], ylim=ylim[[3]])))
  } else {
    return(list(models=models, preds=preds))
  }
}
