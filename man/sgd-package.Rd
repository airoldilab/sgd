\name{sgd-package}
\alias{sgd-package}
\docType{package}
\title{Stochastic gradient descent for parameter estimation of models}
\description{
  This package provides a fast and flexible set of tools for
  estimating parameters in a given model, or more generally a loss function, for
  large scale inference. It includes several stochastic gradient methods, built-in
  models, interval estimation, automated hyperparameter tuning, and convergence
  diagnostics.
}
\author{Tian Lan, Ye Kuang, Panos Toulis, Edoardo Airoldi}
\references{
Toulis, Panos, et al. Statistical analysis of stochastic gradient
methods for generalized linear models.
}
\keyword{ package }

\examples{
## Dobson (1990) Page 93: Randomized Controlled Trial :
counts <- c(18,17,15,20,10,20,25,13,12)
outcome <- gl(3,1,9)
treatment <- gl(3,3)
print(d.AD <- data.frame(treatment, outcome, counts))
sgd.D93 <- sgd(counts ~ outcome + treatment, family = poisson())
print(sgd.D93)

## an example with offsets from Venables & Ripley (2002, p.189)
utils::data(anorexia, package = "MASS")

anorex.1 <- glm(Postwt ~ Prewt + Treat + offset(Prewt),
                family = gaussian, data = anorexia)
}
