\name{sgd.control}
\alias{sgd.control}
\title{Auxiliary for Controlling GLM Fitting}
\description{
  Auxiliary function for \code{\link{sgd}} fitting. Typically only used internally by \code{\link{sgd.fit}}, but may be used to construct a \code{control} argument to either function.
}

\usage{
  sgd.control(epsilon = 1e-08, trace = FALSE, deviance = FALSE, convergence=FALSE)
}
\arguments{
  \item{epsilon}{positive convergence tolerance \eqn{\epsilon}; the iterations converge when \eqn{|dev - dev_{old}|/(|dev| + 0.1) < \epsilon}}

  \item{trace}{logical indicating if output should be produced for each iteration.}

  \item{deviance}{logical indicating if the validity of deviance should be checked in each iteration}

  \item{convergence}{logical indicating if the convergence of the algorithm should be checked.}
}

\details{
  The control argument of \code{\link{sgd}} is by default passed to the \code{control} argument of \code{\link{sgd.fit}}, which uses its elements as arguments to \code{glm.control}: the latter provides defaults and sanity checking.
}

\value{
  A list with components named as the arguments.
}

\examples{
  ## Dobson (1990) Page 93: Randomized Controlled Trial :
  counts <- c(18,17,15,20,10,20,25,13,12)
  outcome <- gl(3,1,9)
  treatment <- gl(3,3)
  print(d.AD <- data.frame(treatment, outcome, counts))
  sgd.D93 <- sgd(counts ~ outcome + treatment, family = poisson(),
   convergence=TRUE, trace=TRUE)
  print(sgd.D93)
}
\author{Tian Lan, Ye Kuang, Panos Toulis, Edoardo Airoldi}

