\name{sgd}
\alias{sgd}
\alias{sgd.formula}
\alias{sgd.function}
\alias{sgd.matrix}
\title{Stochastic gradient descent for parameter estimation of models}
\description{
  This package provides a fast and flexible set of tools for
  estimating parameters in a given model, or more generally a loss function, for
  large scale inference. It includes several stochastic gradient methods, built-in
  models, interval estimation, automated hyperparameter tuning, and convergence
  diagnostics.
}

\usage{
  sgd(x, ...)
  \method{sgd}{formula}(formula, data, model, model.control = list(), sgd.control = list(...), ...)

  \method{sgd}{function}(x, fn.control = list(), sgd.control = list(...), ...)

  \method{sgd}{matrix}(x, y, model, model.control = list(), sgd.control = list(...), ...)
}
\arguments{
  \item{formula}{an object of class "formula" (or one that can be coerced to that class): a symbolic description of the model to be fitted. The details of model specification are given under \code{Details}.}

  \item{data}{an optional data frame, list or environment (or object coercible by \code{\link[base]{as.data.frame}} to a data frame) containing the variables in the model. If not found in data, the variables are taken from environment(formula), typically the environment from which glm is called.}

  \item{model}{character specifying the model to be used: "glm" (generalized linear model).}

  \item{model.control}{a list of parameters for controlling the fitting process. For \code{sgd.fit} this is passed to \code{\link{sgd.control}}.}
  \itemize{
    \item{family ("glm"): }{a description of the error distribution and link function to be used in the model. This can be a character string naming a family function, a family function or the result of a call to a family function. (See \code{\link[stats]{family}} for details of family functions.)}

    \item{intercept ("glm"): }{logical. Should an intercept be included in the \emph{null} model?}
  }

  \item{sgd.control}{a list of parameters for controlling the fitting process. For \code{sgd.fit} this is passed to \code{\link{sgd.control}}.}
  \itemize{
    \item{method: }{character specifying the method to be used: "sgd", "implicit", "asgd"; default is "implicit". See \code{Details}.}

    \item{lr.type: }{character specifying the learning rate to be used: "un-dim", "px-dim", "adagrad"; default is "un-dim". See \code{Details} for their descriptions.}

    \item{start: }{starting values for the parameters in the linear predictor.}

    \item{weights: }{an optional vector of "prior weights" to be used in the fitting process. Should be NULL or a numeric vector.}

    \item{offset: }{this can be used to specify an a priori known component to be included in the linear predictor during fitting. This should be NULL or a numeric vector of length equal to the number of cases. One or more offset terms can be included in the formula instead or as well, and if more than one is specified their sum is used. See \code{\link[stats]{offset}}}
  }

  \item{...}{arguments to be used to form the default \code{sgd.control} arguments if it is not supplied directly.}

  For \code{sgd.function}: x is a function to minimize, and fn.control is a list of controls for x.

  For \code{sgd.matrix}: x is a design matrix of dimension N * d, and y is a vector of observations of length N.
}

\details{
  A typical predictor has the form \code{response ~ terms} where response is the (numeric) response vector and \code{terms} is a series of terms which specifies a linear predictor for \code{response}. For \code{binomial} and \code{quasibinomial} families the response can also be specified as a \code{\link[base]{factor}} (when the first level denotes failure and all others success) or as a two-column matrix with the columns giving the numbers of successes and failures. A terms specification of the form \code{first + second} indicates all the terms in \code{first} together with all the terms in \code{second} with any duplicates removed.

  A specification of the form \code{first:second} indicates the the set of terms obtained by taking the interactions of all terms in \code{first} with all terms in \code{second}. The specification \code{first*second} indicates the \emph{cross} of \code{first} and \code{second}. This is the same as \code{first + second + first:second}.

  The terms in the formula will be re-ordered so that main effects come first, followed by the interactions, all second-order, all third-order and so on: to avoid this pass a \code{terms} object as the formula.

  \code{sgd.matrix} is the workhorse function: it is not normally called directly but can be more efficient where the response vector and design matrix have already been calculated.

  All of \code{weights} and \code{offset} are evaluated in the same way as variables in \code{formula}, that is first in \code{data} and then in the environment of \code{formula}.

  Methods: "sgd" uses stochastic gradient descent (Robbins and Monro, 1951). "implicit" uses implicit stochastic gradient descent (Toulis et al., 2014). "asgd" uses stochastic gradient with averaging (Polyak and Juditsky, 1992).

  Learning rates: "un-dim" uses the one-dimensional learning rate. The method "px-dim" uses the p-dimensional learning rate. The method "adagrad" is more aggressive than "px-dim"
}

\value{
  \code{sgd}
  returns an object of class "sgd".

  An object of class "\code{sgd}" is a list containing at least the following components:

  \code{coefficients}
  a named vector of coefficients

  \code{residuals}
  the \emph{working} residuals, that is the residuals in the final iteration of the fit. Since cases with zero weights are omitted, their working residuals are NA.

  \code{fitted.values}
  the fitted mean values, obtained by transforming the linear predictors by the inverse of the link function.

  \code{rank}
  the numeric rank of the fitted linear model.

  \code{family}
  the \code{\link[stats]{family}} object used.

  \code{linear.predictors}
  the linear fit on link scale.

  \code{deviance}
  up to a constant, minus twice the maximized log-likelihood. Where sensible, the constant is chosen so that a saturated model has deviance zero.

  \code{null.deviance}
  The deviance for the null model, comparable with \code{deviance}. The null model will include the offset, and an intercept if there is one in the model. Note that this will be incorrect if the link function depends on the data other than through the fitted mean: specify a zero offset to force a correct calculation.

  \code{iter}
  the number of iterations of the algorithm used.

  \code{weights}
  the weights initially supplied, a vector of 1s if none were.

  \code{df.residual}
  the residual degrees of freedom.

  \code{df.null}
  the residual degrees of freedom for the null model.

  \code{converged}
  logical. Was the algorithm judged to have converged?

  \code{estimates}
  TODO unknown.
}
\author{Tian Lan, Ye Kuang, Panos Toulis, Edoardo Airoldi}
\references{
  Boris T. Polyak and Anatoli B. Juditsky. Acceleration of stochastic
  approximation by averaging. SIAM Journal on Control and Optimization,
  30(4):838–855, 1992.

  Herbert Robbins and Sutton Monro. A stochastic approximation method. The
  Annals of Mathematical Statistics, pp. 400–407, 1951.

  Panos Toulis, Jason Rennie, and Edoardo M. Airoldi, "Statistical analysis of
  stochastic gradient methods for generalized linear models", In Proceedings of
  the 31st International Conference on Machine Learning (ICML), 2014.
}
\examples{
  ## Dobson (1990, p.93): Randomized Controlled Trial
  counts <- c(18, 17, 15, 20, 10, 20, 25, 13, 12)
  outcome <- gl(3, 1, 9)
  treatment <- gl(3, 3)
  print(d.AD <- data.frame(treatment, outcome, counts))
  sgd.D93 <- sgd(counts ~ outcome + treatment, model="glm",
                 model.control=list(family = poisson()))
  sgd.D93

  ## Venables & Ripley (2002, p.189): an example with offsets
  utils::data(anorexia, package="MASS")

  anorex.1 <- sgd(Postwt ~ Prewt + Treat + offset(Prewt),
                  family=gaussian, data=anorexia)
}
