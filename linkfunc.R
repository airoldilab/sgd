link_normal <- function(x) {
  # link function h(.) for normal distribution
  return(x)
}

link_logistic <- function(x) {
  # link function for logistic distribution
  return(exp(x) / (1+exp(x)))
}

link_poisson <- function(x) {
  # link function for poisson distribution
  return(exp(x))
}