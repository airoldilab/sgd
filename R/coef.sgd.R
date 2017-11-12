#' Extract Model Coefficients
#'
#' Extract model coefficients from \code{sgd} objects. \code{coefficients}
#' is an \emph{alias} for it.
#'
#' @param object object of class \code{sgd}.
#' @param \dots some methods for this generic require additional
#'   arguments. None are used in this method.
#'
#' @return
#' Coefficients extracted from the model object \code{object}.
#'
#' @export
coef.sgd <- function(object, ...) {
  return(as.vector(object$coefficients))
}
