#' Print objects of class \code{sgd}.
#'
#' @param x object of class \code{sgd}.
#' @param \dots further arguments passed to or from other methods.
#'
#' @export
print.sgd <- function(x, ...) {
  print(coef(x), ...)
}
