#' Predict for objects of class \code{sgd}
#'
#' Form predictions using the estimated model parameters from stochastic
#' gradient descent.
#'
#' @param x object of class \code{sgd}.
#' @param x_test design matrix to form predictions on
#' @param \dots further arguments passed to or from other methods.
#'
#' @details
#' A column of 1's must be included if the parameters include a bias, or
#' intercept, term.
#'
#' @export
predict.sgd <- function(x, x_test, ...) {
  if (x$model %in% c("lm", "glm")) {
    eta <- x_test %*% x$coefficients
    y <- x$model.out$family$linkinv(eta)
  } else if(x$model == "m") {
    eta <- x_test %*% x$estimates
    y <- eta
  # TODO
  } else {
    stop("'model' not recognized")
  }
  return(y)
}

#' @export
#' @rdname predict.sgd
predict_all <- function(x, x_test, ...) {
  if (x$model %in% c("lm", "glm")) {
    eta <- x_test %*% x$estimates
    y <- x$model.out$family$linkinv(eta)
  } else if(x$model == "m") {
    eta <- x_test %*% x$estimates
    y <- eta
  # TODO
  } else {
    stop("'model' not recognized")
  }
  return(y)
}
