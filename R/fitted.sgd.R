#' Extract Model Fitted Values
#'
#' Extract fitted values from from \code{sgd} objects.
#' \code{fitted.values} is an \emph{alias} for it.
#'
#' @param object object of class \code{sgd}.
#' @param \dots some methods for this generic require additional
#'   arguments. None are used in this method.
#'
#' @return
#' Fitted values extracted from the object \code{object}.
#'
#' @export
fitted.sgd <- function(object, ...) {
  return(object$fitted.values)
}
