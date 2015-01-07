\name{implicit}
\alias{implicit.fit}
\alias{implicit.formula}
\alias{implicit}
\title{Fit Generalized Linear Models}
\description{
  \code{implicit} is used to fit generalized linear models, 
 specified by giving a symbolic description of the linear predictor and
  a description of the error distribution. Stochastic Gradient Descent (SGD),
  averaged-SGD (ASGD), implicit-SGD can be used for model fitting.
}

\usage{
  implicit(x, ...)
  \method{implicit}{formula}(formula, family = gaussian, data, weights, subset, 
           na.action, start = NULL, offset, control = list(...), 
           model = TRUE, method = "implicit", x = FALSE, y = TRUE, contrasts = NULL, 
           lr.type = "uni-dim", ...)

  \method{implicit}{fit}(x, y, weights = rep(1, nobs), start = NULL,
               offset = rep(0, nobs), family = gaussian(), 
               control = list(), intercept = TRUE, method="implicit", lr.type, ...)
}
\arguments{
  \item{formula}{an object of class "formula" (or one that can be coerced to that class): a symbolic description of the model to be fitted. The details of model specification are given under "\code{Details}".}

  \item{family}{a description of the error distribution and link function to be used in the model. This can be a character string naming a family function, a family function or the result of a call to a family function. (See \code{\link[stats]{family}} for details of family functions.)}

  \item{data}{an optional data frame, list or environment (or object coercible by \code{\link[base]{as.data.frame}} to a data frame) containing the variables in the model. If not found in data, the variables are taken from environment(formula), typically the environment from which glm is called.}

  \item{weights}{an optional vector of "prior weights" to be used in the fitting process. Should be NULL or a numeric vector.}

  \item{subset}{an optional vector specifying a subset of observations to be used in the fitting process.}

  \item{na.action}{a function which indicates what should happen when the data contain NAs. The default is set by the na.action setting of options, and is na.fail if that is unset. The "factory-fresh" default is na.omit. Another possible value is NULL, no action. Value na.exclude can be useful.}

  \item{start}{starting values for the parameters in the linear predictor.}

  \item{offset}{this can be used to specify an a priori known component to be included in the linear predictor during fitting. This should be NULL or a numeric vector of length equal to the number of cases. One or more offset terms can be included in the formula instead or as well, and if more than one is specified their sum is used. See \code{\link[stats]{offset}}}

  \item{control}{a list of parameters for controlling the fitting process. For \code{implicit.fit} this is passed to \code{\link{implicit.control}}.}

  \item{model}{a logical value indicating whether model frame should be included as a component of the returned value.}

  \item{method}{the method to be used in fitting the model. The default method "implicit" uses implicit-SGD. Other possible methodes include "asgd" and "sgd". The alternative method "model.frame" returns the model frame and does no fitting.}

  \item{x, y}{For \code{implicit}: logical values indicating whether the response vector and model matrix used in the fitting process should be returned as components of the returned value.}

  For \code{implicit.fit}: x is a design matrix of dimension n * p, and y is a vector of observations of length n.

  \item{contrasts}{an optional list. See the \code{contrasts.arg} of \code{model.matrix.default}.}

  \item{intercept}{logical. Should an intercept be included in the \emph{null} model?}

  \item{lr.type}{the type of learning rate to be used for fitting the model. The default method "un-dim" uses the one-dimensional learning rate. The alternative method "px-dim" uses the p-dimensional learning rate.}

  \item{...}{arguments to be used to form the default control argument if it is not supplied directly.}
}

\details{
  A typical predictor has the form \code{response ~ terms} where response is the (numeric) response vector and \code{terms} is a series of terms which specifies a linear predictor for \code{response}. For \code{binomial} and \code{quasibinomial} families the response can also be specified as a \code{\link[base]{factor}} (when the first level denotes failure and all others success) or as a two-column matrix with the columns giving the numbers of successes and failures. A terms specification of the form \code{first + second} indicates all the terms in \code{first} together with all the terms in \code{second} with any duplicates removed.

  A specification of the form \code{first:second} indicates the the set of terms obtained by taking the interactions of all terms in \code{first} with all terms in \code{second}. The specification \code{first*second} indicates the \emph{cross} of \code{first} and \code{second}. This is the same as \code{first + second + first:second}.

  The terms in the formula will be re-ordered so that main effects come first, followed by the interactions, all second-order, all third-order and so on: to avoid this pass a \code{terms} object as the formula.

  Non-\code{NULL weights} can be used to indicate that different observations have different dispersions (with the values in \code{weights} being inversely proportional to the dispersions); or equivalently, when the elements of \code{weights} are positive integers \code{w_i}, that each response \code{y_i} is the mean of \code{w_i} unit-weight observations. For a binomial GLM prior weights are used to give the number of trials when the response is the proportion of successes: they would rarely be used for a Poisson GLM.

  \code{implicit.fit} is the workhorse function: it is not normally called directly but can be more efficient where the response vector and design matrix have already been calculated.

  All of \code{weights}, \code{subset}, and \code{offset} are evaluated in the same way as variables in \code{formula}, that is first in \code{data} and then in the environment of \code{formula}.

  For the background to warning messages about "fitted probabilities numerically 0 or 1 occurred" for binomial GLMs, see Venables & Ripley (2002, pp. 197-8).
}

\value{
  \code{implicit} 
  returns an object of class inheriting from "implicit" which inherits from the class "\code{lm}" and "\code{glm}". 

  The function \code{\link[base]{summary}} (i.e., \code{summary.glm}) can be used to obtain or print a summary of the results and the function \code{\link[stats]{anova}} (i.e., \code{anova.glm}) to produce an analysis of variance table.

  The generic accessor functions \code{\link[stats]{coefficients}}, \code{\link[stats]{fitted.values}} and \code{\link[stats]{residuals}} can be used to extract various useful features of the value returned by \code{implicit}.

  \code{weights} 
  extracts a vector of weights, one for each case in the fit (after subsetting and na.action).

  An object of class "\code{implicit}" is a list containing at least the following components:

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

  \code{aic} 
  A version of \emph{Akaike's An Information Criterion}, minus twice the maximized log-likelihood plus twice the number of parameters, computed by the \code{aic} component of the family. For binomial and Poison families the dispersion is fixed at one and the number of parameters is the number of coefficients. For gaussian, Gamma and inverse gaussian families the dispersion is estimated from the residual deviance, and the number of parameters is the number of coefficients plus one. For a gaussian family the MLE of the dispersion is used so this is a valid value of AIC, but for Gamma and inverse gaussian families it is not. For families fitted by quasi-likelihood the value is NA.

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

  \code{y} 
  if requested (the default) the \code{y} vector used. (It is a vector even for a binomial model.)

  \code{x} 
  if requested, the model matrix.

  \code{model} 
  if requested (the default), the model frame.

  \code{converged} 
  logical. Was the algorithm judged to have converged?

  \code{call}  
  the matched call.

  \code{formula} 
  the formula supplied.

  \code{terms} 
  the \code{\link[stats]{terms}} object used.

  \code{data}  
  the data \code{argument}.

  \code{offset}  
  the offset vector used.

  \code{control}  
  the value of the \code{control} argument used.

  \code{method}  
  the name of the fitter function used.

  \code{contrasts} 
  (where relevant) the contrasts used.

  \code{xlevels} 
  (where relevant) a record of the levels of the factors used in fitting.

  \code{na.action} 
  (where relevant) information returned by \code{\link[stats]{model.frame}} on the special handling of NAs.

  In addition, non-empty fits will have components \code{qr} and \code{R} relating to the final  linear fit.

  Objects of class "\code{implicit}" are normally of class c("\code{glm}", "\code{lm}", "\code{implicit}"), that is inherit from class "\code{lm}" and "\code{glm}", and well-designed methods for class "\code{lm}" and "\code{glm}" will be applied to the model at the final iteration.

}

\references{
  Toulis, Panos, et al. Statistical analysis of stochastic gradient 
methods for generalized linear models.
}
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
\author{Tian Lan, Ye Kuang, Panos Toulis, Edoardo Airoldi}

