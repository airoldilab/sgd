source("load_mnist.R")
source("multilogit.R")
source("plot.R")


run_exp <- function(methods, lrs, np, X, y, X_test, y_test, plot=T){
  models = list()
  preds = list()
  y_tests = list()
  for (i in 1:length(methods)){
    model <- multilogit.fit(X, y, sgd.control=list(
      method=methods[[i]], lr=lrs[[i]], npasses=np[[i]]))
    pred <- multilogit.predict(model, X_test) 
    models[[i]] <- model
    preds[[i]] <- pred
    y_tests[[i]] <- y_test
    print(sprintf("%d experiment done!", i))
  }
  if (plot){
    return(plot.error(preds, y_tests, methods))
  } else{
    return(list(models=models, preds=preds))
  }
}

dat <- load_mnist()
X <- dat$train$x
y <- dat$train$y
X_test <- dat$test$x
y_test <- dat$test$y


methods <- list("sgd", "ai-sgd", "implicit")
lrs <- list("adagrad", "one-dim", "one-dim")
np <- list(3, 3, 3)

methods <- list("ai-sgd")
lrs <- list("one-dim")
np <- list(4)
run_exp(methods, lrs, np, X, y, X_test, y_test)
