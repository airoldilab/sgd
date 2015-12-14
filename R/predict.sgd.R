#' Predict for objects of class \code{sgd}
#'
#' Form predictions using the estimated model parameters from stochastic
#' gradient descent.
#'
#' @param object object of class \code{sgd}.
#' @param x_test design matrix to form predictions on
#' @param \dots further arguments passed to or from other methods.
#'
#' @details
#' A column of 1's must be included if the parameters include a bias, or
#' intercept, term.
#'
#' @export
predict.sgd <- function(object, x_test, ...) {
  if (object$model %in% c("lm", "glm")) {
    eta <- x_test %*% object$coefficients
    y <- object$model.out$family$linkinv(eta)
  } else if (object$model == "m") {
    eta <- x_test %*% object$estimates
    y <- eta
  # TODO
  } else {
    stop("'model' not recognized")
  }
  return(y)
}

#' @export
#' @rdname predict.sgd
predict_all <- function(object, x_test, ...) {
  if (object$model %in% c("lm", "glm")) {
    eta <- x_test %*% object$estimates
    y <- object$model.out$family$linkinv(eta)
  } else if (object$model == "m") {
    eta <- x_test %*% object$estimates
    y <- eta
  # TODO
  } else {
    stop("'model' not recognized")
  }
  return(y)
}
