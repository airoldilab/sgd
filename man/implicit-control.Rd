\name{implicit.control}
\alias{implicit.control}
\title{Auxiliary for Controlling GLM Fitting}
\description{
  Auxiliary function for \code{\link{implicit}} fitting. Typically only used internally by \code{\link{implicit.fit}}, but may be used to construct a \code{control} argument to either function.
}

\usage{
  implicit.control(epsilon = 1e-08, trace = FALSE, deviance = FALSE, convergence=FALSE) 
}
\arguments{
  \item{epsilon}{positive convergence tolerance \eqn{\epsilon}; the iterations converge when \eqn{|dev - dev_{old}|/(|dev| + 0.1) < \epsilon}}

  \item{trace}{logical indicating if output should be produced for each iteration.}

  \item{deviance}{logical indicating if the validity of deviance should be checked in each iteration}

  \item{convergence}{logical indicating if the convergence of the algorithm should be checked.}
}

\details{
  The control argument of \code{\link{implicit}} is by default passed to the \code{control} argument of \code{\link{implicit.fit}}, which uses its elements as arguments to \code{glm.control}: the latter provides defaults and sanity checking.
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
  implicit.D93 <- implicit(counts ~ outcome + treatment, family = poisson(),
   convergence=TRUE, trace=TRUE)
  print(implicit.D93)
}
\author{Tian Lan, Ye Kuang, Panos Toulis, Edoardo Airoldi}

