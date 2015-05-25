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
    ptm <- proc.time()
    X_temp <- X[y==label | y==pivot.label, ]
    y_temp <- y[y==label | y==pivot.label]
    select <- y_temp==label
    y_temp[!select] <- 0
    y_temp[select] <- 1
    X_temp <- cbind(rep(1, nrow(X_temp)), X_temp)
    model <- sgd(X_temp, y_temp, "glm", model.control=list(family="binomial"), ...)
    coefs[count, , ] <- model$estimates
    pos[count-1, ] <- model$pos
    time <-proc.time()[3]-ptm[3]
    print(sprintf("Finish fitting %d out of %d labels; Time: %f s", count-1, nlabels-1, time))
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

run_exp <- function(methods, names, lrs, np, X, y, X_test, y_test, plot=T){
  
  # Args: 
  #  methods: a list of "sgd", "implicit" or "ai-sgd"
  #  names: a list of labels for each experiment for plotting
  #  lrs: a list of learning rate types
  #  np: a list of number of passes
  
  models = list()
  preds = list()
  y_tests = list()
  for (i in 1:length(methods)){
    ptm <- proc.time()
    model <- multilogit.fit(X, y, sgd.control=list(
      method=methods[[i]], lr=lrs[[i]], npasses=np[[i]]))
    pred <- multilogit.predict(model, X_test) 
    models[[i]] <- model
    preds[[i]] <- pred
    y_tests[[i]] <- y_test
    time <- proc.time()[3]-ptm[3]
    print(sprintf("experiment %d of %d done! Time: %f s", i, length(methods), time))
  }
  if (plot){
    return(plot.error(preds, y_tests, names))
  } else{
    return(list(models=models, preds=preds))
  }
}
