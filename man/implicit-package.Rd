\name{implicit-package}
\alias{implicit-package}
\docType{package}
\title{Implicit SGD for Fitting Generalized Linear Models}
\description{\code{implicit} is used to fit generalized linear models, 
 specified by giving a symbolic description of the linear predictor and
  a description of the error distribution. Stochastic Gradient Descent (SGD),
  averaged-SGD (ASGD), implicit-SGD can be used for model fitting.}
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
implicit.D93 <- implicit(counts ~ outcome + treatment, family = poisson())
print(implicit.D93)

## an example with offsets from Venables & Ripley (2002, p.189)
utils::data(anorexia, package = "MASS")

anorex.1 <- glm(Postwt ~ Prewt + Treat + offset(Prewt),
                family = gaussian, data = anorexia)
}
