#' Extract Model Residuals
#'
#' Extract model residuals from \code{sgd} objects. \code{resid} is an
#' \emph{alias} for it.
#'
#' @param object object of class \code{sgd}.
#' @param \dots some methods for this generic require additional
#'   arguments. None are used in this method.
#'
#' @return
#' Residuals extracted from the object \code{object}.
#'
#' @export
residuals.sgd <- function(object, ...) {
  return(object$residuals)
}
