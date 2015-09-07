#' Predict for objects of class \code{sgd}
#'
#' Form predictions using the estimated model parameters from stochastic
#' gradient descent.
#'
#' @param x object of class \code{sgd}.
#' @param x_test design matrix to form predictions on
#' @param \dots further arguments passed to or from other methods.
#'
#' @export
predict.sgd <- function(x, x_test, ...) {
  # TODO
  #if (x$model == "cox") {
  #} else if (x$model == "gmm") {
  #} else if (x$model %in% c("lm", "glm")) {
  if (x$model %in% c("lm", "glm")) {
    eta <- x_test %*% x$coefficients # assuming intercepts in X
    y <- x$model.out$family$linkinv(eta)
  } else {
    stop("'model' not recognized")
  }
  return(y)
}
