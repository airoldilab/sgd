#' Model Predictions
#'
#' Form predictions using the estimated model parameters from stochastic
#' gradient descent.
#'
#' @param object object of class \code{sgd}.
#' @param newdata design matrix to form predictions on
#' @param type the type of prediction required. The default "link" is
#'   on the scale of the linear predictors; the alternative '"response"'
#'   is on the scale of the response variable. Thus for a default
#'   binomial model the default predictions are of log-odds
#'   (probabilities on logit scale) and 'type = "response"' gives the
#'   predicted probabilities. The '"terms"' option returns a matrix
#'   giving the fitted values of each term in the model formula on the
#'   linear predictor scale.
#' @param \dots further arguments passed to or from other methods.
#'
#' @details
#' A column of 1's must be included to \code{newdata} if the
#' parameters include a bias (intercept) term.
#'
#' @export
predict.sgd <- function(object, newdata, type="link", ...) {
  if (!(object$model %in% c("lm", "glm", "m"))) {
    stop("'model' not supported")
  }
  if (!(type %in% c("link", "response", "term"))) {
    stop("'type' not recognized")
  }

  if (object$model %in% c("lm", "glm")) {
    if (type %in% c("link", "response")) {
      eta <- newdata %*% coef(object)
      if (type == "response") {
        y <- object$model.out$family$linkinv(eta)
        return(y)
      }
      return(eta)
    }
    eta <- newdata %*% diag(coef(object))
    return(eta)
  } else if (object$model == "m") {
    if (type %in% c("link", "response")) {
      eta <- newdata %*% coef(object)
      if (type == "response") {
        y <- eta
        return(y)
      }
      return(eta)
    }
    eta <- newdata %*% diag(coef(object))
    return(eta)
  }
}

#' @export
#' @rdname predict.sgd
predict_all <- function(object, newdata, ...) {
  if (object$model %in% c("lm", "glm")) {
    eta <- newdata %*% object$estimates
    y <- object$model.out$family$linkinv(eta)
  } else if (object$model == "m") {
    eta <- newdata %*% object$estimates
    y <- eta
  # TODO
  } else {
    stop("'model' not recognized")
  }
  return(y)
}
