#' Stochastic gradient descent
#'
#' Run stochastic gradient descent on the underlying loss function for a given
#' model and data, or a user-specified loss function.
#'
#' @param formula an object of class \code{"\link{formula}"} (or one that can be
#'   coerced to that class): a symbolic description of the model to be fitted.
#'   The details of model specification are given under \sQuote{Details}.
#' @param data an optional data frame, list or environment (or object coercible
#'   by \code{\link[base]{as.data.frame}} to a data frame) containing the
#'   variables in the model. If not found in data, the variables are taken from
#'   environment(formula), typically the environment from which glm is called.
#' @param model character specifying the model to be used: \code{"glm"}
#'   (generalized linear model).
#' @param model.control a list of parameters for controlling the model.
#'   \itemize{
#'     \item family (\code{"glm"}): a description of the error distribution and
#'       link function to be used in the model. This can be a character string
#'       naming a family function, a family function or the result of a call to
#'       a family function.  (See \code{\link[stats]{family}} for details of
#'       family functions.)
#'     \item intercept (\code{"glm"}): logical. Should an intercept be included
#'       in the \emph{null} model?
#'   }
#' @param sgd.control a list of parameters for controlling the estimation
#'   \itemize{
#'     \item method: character specifying the method to be used: \code{"sgd"},
#'       \code{"implicit"}, \code{"asgd"}. Default is \code{"implicit"}. See
#'       \sQuote{Details}.
#'     \item lr.type: character specifying the learning rate to be used:
#'       \code{"uni-dim"}, \code{"uni-dim-eigen"}, \code{"p-dim"},
#'       \code{"p-dim-weighted"}, \code{"adagrad"}. Default is \code{"uni-dim"}.
#'       See \sQuote{Details}.
#'     \item start: starting values for the parameter estimates. Default is
#'       random initialization around the mean.
#'     \item weights: an optional vector of "prior weights" to be used in the
#'       fitting process. Should be NULL or a numeric vector.
#'     \item offset: this can be used to specify an a priori known component to
#'       be included in the linear predictor during fitting. This should be NULL
#'       or a numeric vector of length equal to the number of cases. One or more
#'       offset terms can be included in the formula instead or as well, and if
#'       more than one is specified their sum is used. See
#'       \code{\link[stats]{offset}}
#'   }
#' @param \dots arguments to be used to form the default \code{sgd.control}
#'   arguments if it is not supplied directly.
#'
#' For \code{sgd.function}: x is a function to minimize, and fn.control is a
#' list of controls for x.
#'
#' For \code{sgd.matrix}: x is a design matrix of dimension N * d, and y is a
#' vector of observations of length N.
#'
#' @details
#' A typical predictor has the form \code{response ~ terms} where response is
#' the (numeric) response vector and \code{terms} is a series of terms which
#' specifies a linear predictor for \code{response}.  For \code{binomial} and
#' \code{quasibinomial} families the response can also be specified as a
#' \code{\link[base]{factor}} (when the first level denotes failure and all
#' others success) or as a two-column matrix with the columns giving the
#' numbers of successes and failures.  A terms specification of the form
#' \code{first + second} indicates all the terms in \code{first} together with
#' all the terms in \code{second} with any duplicates removed.
#'
#' A specification of the form \code{first:second} indicates the the set of
#' terms obtained by taking the interactions of all terms in \code{first} with
#' all terms in \code{second}.  The specification \code{first*second} indicates
#' the \emph{cross} of \code{first} and \code{second}.  This is the same as
#' \code{first + second + first:second}.
#'
#' The terms in the formula will be re-ordered so that main effects come first,
#' followed by the interactions, all second-order, all third-order and so on:
#' to avoid this pass a \code{terms} object as the formula.
#'
#' \code{sgd.matrix} is the workhorse function: it is not normally called
#' directly but can be more efficient where the response vector and design
#' matrix have already been calculated.
#'
#' All of \code{weights} and \code{offset} are evaluated in the same way as
#' variables in \code{formula}, that is first in \code{data} and then in the
#' environment of \code{formula}.
#'
#' Methods: "sgd" uses stochastic gradient descent (Robbins and Monro, 1951).
#' "implicit" uses implicit stochastic gradient descent (Toulis et al., 2014).
#' "asgd" uses stochastic gradient with averaging (Polyak and Juditsky, 1992).
#'
#' Learning rates: "uni-dim" uses the one-dimensional learning rate.  The
#' method "p-dim" uses the p-dimensional learning rate.  The method "adagrad"
#' uses a diagonal scaling (Duchi et al., 2011).
#'
#' @return
#' An object of class \code{"sgd"}, which is a list containing at least the
#' following components:
#'
#' \code{coefficients}
#' a named vector of coefficients
#'
#' \code{residuals}
#' the \emph{working} residuals, that is the residuals in the final iteration of
#' the fit. Since cases with zero weights are omitted, their working residuals
#' are NA.
#'
#' \code{fitted.values}
#' the fitted mean values, obtained by transforming the linear predictors by the
#' inverse of the link function.
#'
#' \code{rank}
#' the numeric rank of the fitted linear model.
#'
#' \code{family}
#' the \code{\link[stats]{family}} object used.
#'
#' \code{linear.predictors}
#' the linear fit on link scale.
#'
#' \code{deviance}
#' up to a constant, minus twice the maximized log-likelihood. Where sensible,
#' the constant is chosen so that a saturated model has deviance zero.
#'
#' \code{null.deviance}
#' The deviance for the null model, comparable with \code{deviance}. The null
#' model will include the offset, and an intercept if there is one in the model.
#' Note that this will be incorrect if the link function depends on the data
#' other than through the fitted mean: specify a zero offset to force a correct
#' calculation.
#'
#' \code{iter}
#' the number of iterations of the algorithm used.
#'
#' \code{weights}
#' the weights initially supplied, a vector of 1s if none were.
#'
#' \code{df.residual}
#' the residual degrees of freedom.
#'
#' \code{df.null}
#' the residual degrees of freedom for the null model.
#'
#' \code{converged}
#' logical. Was the algorithm judged to have converged?
#'
#' \code{estimates}
#' TODO unknown.
#'
#' @author Dustin Tran, Tian Lan, Panos Toulis, Ye Kuang, Edoardo Airoldi
#' @references
#' John Duchi, Elad Hazan, and Yoram Singer. Adaptive subgradient methods for
#' online learning and stochastic optimization. \emph{Journal of Machine
#' Learning Research}, 12:2121–2159, 2011.
#'
#' Boris T. Polyak and Anatoli B. Juditsky. Acceleration of stochastic
#' approximation by averaging. \emph{SIAM Journal on Control and Optimization},
#' 30(4):838–855, 1992.
#'
#' Herbert Robbins and Sutton Monro. A stochastic approximation method.
#' \emph{The Annals of Mathematical Statistics}, pp. 400–407, 1951.
#'
#' Panos Toulis, Jason Rennie, and Edoardo M. Airoldi, "Statistical analysis of
#' stochastic gradient methods for generalized linear models", In
#' \emph{Proceedings of the 31st International Conference on Machine Learning},
#' 2014.
#'
#' @examples
#' ## Dobson (1990, p.93): Randomized Controlled Trial
#' counts <- c(18, 17, 15, 20, 10, 20, 25, 13, 12)
#' outcome <- gl(3, 1, 9)
#' treatment <- gl(3, 3)
#' print(d.AD <- data.frame(treatment, outcome, counts))
#' sgd.D93 <- sgd(counts ~ outcome + treatment, model="glm",
#'          model.control=list(family = poisson()))
#' sgd.D93
#'
#' ## Venables & Ripley (2002, p.189): an example with offsets
#' utils::data(anorexia, package="MASS")
#'
#' anorex.1 <- sgd(Postwt ~ Prewt + Treat + offset(Prewt),
#'                 family=gaussian, data=anorexia)
#'
#' @useDynLib sgd
#' @import MASS
#' @importFrom Rcpp evalCpp
#' @aliases sgd.formula sgd.function sgd.matrix
################################################################################
# Classes
################################################################################
#' @export
sgd <- function(x, ...) UseMethod("sgd")

################################################################################
# Methods
################################################################################

#' @export
sgd.default <- function(x, ...) {
  stop("class of x is not a formula, function, or matrix")
}

#' @export
#' @rdname sgd
sgd.formula <- function(formula, data, model,
                        model.control=list(),
                        sgd.control=list(...),
                        ...) {
  # TODO
  # subset: a subset of data points; can be a parameter in sgd.control
  # na.action: how to deal when data has NA; can be a parameter in sgd.control
  # model: logical value determining whether to output the X data frame
  # x,y: logical value determining whether to output the x and/or y
  # contrasts: a list for performing hypothesis testing on other sets of predictors; can be a paramter in sgd.control
  # Call method when the first argument is a formula
  # the call parameter to return
  call <- match.call()

  # 1. Validity check.
  if (missing(model)) {
    stop("model not specified")
  }
  if (missing(data)) {
    data <- environment(formula)
  }

  # 2. Build dataframe according to the formula.
  mf <- match.call(expand.dots=FALSE)
  m <- match(c("formula", "data"), names(mf), 0L)
  mf <- mf[c(1L, m)]
  mf$drop.unused.levels
  mf[[1L]] <- quote(stats::model.frame)
  mf <- eval(mf, parent.frame())

  Y <- model.response(mf, "any")
  if (length(dim(Y)) == 1L) {
    nm <- rownames(Y)
    dim(Y) <- NULL
    if (!is.null(nm)) {
      names(Y) <- nm
    }
  }

  mt <- attr(mf, "terms")
  if (!is.empty.model(mt)) {
    X <- model.matrix(mt, mf)
  } else {
    X <- matrix(, NROW(Y), 0L)
  }

  # 3. Pass into sgd.matrix().
  return(sgd.matrix(X, Y, model, model.control, sgd.control))
}

#' @export
#' @rdname sgd
sgd.function <- function(x,
                        fn.control=list(),
                        sgd.control=list(...),
                        ...) {
  # TODO run_online_algorithm will not work on this as it relies on data
  # sgd.fn.control.valid
  gr <- NULL
  lower <- -Inf
  upper <- Inf
}

#' @export
#' @rdname sgd
sgd.matrix <- function(x, y, model,
                       model.control=list(),
                       sgd.control=list(...),
                       ...) {
  # Call method when the first argument is a formula
  # the call parameter to return
  call <- match.call()

  # 1. Validity check.
  if (missing(x)) {
    stop("x not specified")
  }
  if (missing(y)) {
    stop("y not specified")
  }
  if (missing(model)) {
    stop("model not specified")
  }
  if (!is.list(model.control))  {
    stop("'model.control' is not a list")
  } else {
    model.control <- do.call("sgd.model.control.valid", c(model.control,
                             model=model))
  }
  if (!is.list(sgd.control))  {
    stop("'sgd.control' is not a list")
  }
  sgd.control <- do.call("sgd.sgd.control.valid", sgd.control)

  # 2. Fit!
  if (model == "glm") {
    fit <- sgd.fit.glm
  } else {
    print(model)
    stop("'model' not recognized")
  }
  out <- fit(x, y, model.control, sgd.control)
  class(out) <- c(out$class, "sgd")
  return(out)
}

################################################################################
# Generic methods
################################################################################

# TODO
#print.sgd <- function(x) {
#  # What goes to standard output.
#  #
#  # Args:
#  #   x:    sgd object
#}
#plot.sgd <- function(x, type="mse") {
#  # An all-encompassing visualization routine.
#  #
#  # Args:
#  #   x:    sgd object
#  #   type: character in c("")
#  #
#  # Returns:
#  #   A plot.
#  if (type == "mse") {
#    plot <- plot.sgd.mse
#  } else {
#    print(type)
#    stop("'type' not recognized")
#  }
#  return(plot(x))
#}

################################################################################
# Auxiliary functions: model fitting
################################################################################

sgd.fit.glm <- function(x, y,
                        model.control,
                        sgd.control) {
  xnames <- dimnames(x)[[2L]]
  ynames <- ifelse(is.matrix(y), rownames(y), names(y))
  N <- NROW(y) # number of observations
  d <- ncol(x) # number of features
  EMPTY <- d == 0

  family <- model.control$family
  intercept <- model.control$intercept

  start <- sgd.control$start
  method <- sgd.control$method
  lr.type <- sgd.control$lr.type
  if (is.null(sgd.control$weights)) {
    weights <- rep.int(1, N)
  } else {
    weights <- sgd.control$weights
  }
  if (is.null(sgd.control$offset)) {
    offset <- rep.int(0, N)
  } else {
    offset <- sgd.control$offset
  }
  implicit.control <- do.call("sgd.implicit.control", sgd.control)

  variance <- family$variance
  linkinv <- family$linkinv
  if (!is.function(variance) || !is.function(linkinv)) {
    stop("'family' argument seems not to be a valid family object",
         call.=FALSE)
  }
  dev.resids <- family$dev.resids
  mu.eta <- family$mu.eta

  unless.null <- function(x, if.null) ifelse(is.null(x), if.null, x)
  valideta <- unless.null(family$valideta, function(eta) TRUE)
  validmu <- unless.null(family$validmu, function(mu) TRUE)

  if (EMPTY) {
    eta <- rep.int(0, N) + offset
    if (!valideta(eta)) {
      stop("invalid linear predictor values in empty model",
           call.=FALSE)
    }
    mu <- linkinv(eta)
    if (!validmu(mu)) {
      stop("invalid fitted means in empty model", call.=FALSE)
    }
    dev <- sum(dev.resids(y, mu, weights))
    w <- ((weights * mu.eta(eta)^2)/variance(mu))^0.5
    residuals <- (y - mu)/mu.eta(eta)
    good <- rep_len(TRUE, length(residuals))
    boundary <- conv <- TRUE
    coef <- numeric()
    iter <- 0L
    rank <- 0L
    converged <- FALSE
  } else {
    # Set the initial value for theta.
    if (!is.null(start) & length(start) != d) {
      stop(gettextf("length of 'start' should equal %d and correspond to initial coefs for %s",
                    d, paste(deparse(xnames), collapse=", ")), domain=NA)
    } else {
      start <- rep(0, d)
    }
    eta <- sum(x[1, ] * start)+offset[1]
    if (!valideta(eta)) {
      stop("cannot find valid starting values: please specify some", call.=FALSE)
    }
    y <- as.matrix(y)

    # Select x, y with weights > 0; adjust for offsets.
    good <- weights > 0
    dataset <- list(X=as.matrix(x[good, ]), Y=as.matrix(y[good]))
    experiment <- list()
    experiment$name <- family$family
    experiment$model.attrs <- list()
    experiment$model.attrs$transfer.name <- sgd.transfer.name(family$link)
    experiment$niters <- length(dataset$Y)
    experiment$lr.type <- lr.type
    experiment$p <- dim(dataset$X)[2]
    experiment$weights <- as.matrix(weights[good])
    experiment$start <- as.matrix(start)
    experiment$deviance <- implicit.control$deviance
    experiment$trace <- implicit.control$trace
    experiment$convergence <- implicit.control$convergence
    experiment$epsilon <- implicit.control$epsilon
    experiment$offset <- as.matrix(offset[good])
    out <- run_online_algorithm(dataset, experiment, method, verbose=F)
    if (length(out) == 0) {
      stop("An error has occured, program stopped.")
    }
    temp.mu <- as.numeric(out$mu)
    mu <- rep(0, length(good))
    mu[good] <- temp.mu
    mu[!good] <- NA
    temp.eta <- as.numeric(out$eta)
    eta <- rep(0, length(good))
    eta[good] <- temp.eta
    eta[!good] <- NA
    coef <- as.numeric(out$coefficients)
    dev <- out$deviance
    residuals <- as.numeric((y - mu)/mu.eta(eta))
    iter <- experiment$p
    rank <- out$rank
    converged <- out$converged
  }
  names(residuals) <- ynames
  names(mu) <- ynames
  names(eta) <- ynames
  names(weights) <- ynames
  names(y) <- ynames
  wtdmu <- ifelse(intercept, sum(weights * y)/sum(weights), linkinv(offset))
  nulldev <- sum(dev.resids(y, wtdmu, weights))
  n.ok <- N - sum(weights == 0)
  nulldf <- n.ok - as.integer(intercept)
  resdf <- n.ok - rank
  names(coef) <- xnames
  return(list(coefficients=coef,
       residuals=residuals,
       fitted.values=mu,
       rank=rank,
       family=family,
       linear.predictors=eta,
       deviance=dev,
       null.deviance=nulldev,
       iter=iter,
       weights=weights,
       df.residual=resdf,
       df.null=nulldf,
       estimates=if(!EMPTY) out$estimates,
       converged=if(implicit.control$convergence) converged))
  # TODO in C: deal with offset
  # TODO compare all results with glm
  # TODO unit test on all checks
  # TODO write start value
}

sgd.implicit.control <- function(epsilon=1e-08, trace=FALSE, deviance=FALSE,
                                 convergence=FALSE, ...) {
  # Maintain control parameters for running implicit SGD.
  #
  # Args:
  #   epsilon:     positive convergence tolerance; the iterations
  #                converge when |dev - dev_{old}|/(|dev| + 0.1) < epsilon
  #   trace:       logical indicating if output should be produced for each
  #                iteration
  #   deviance:    logical indicating if the validity of deviance should be
  #                checked in each iteration
  #   convergence: logical indicating if the convergence of the algorithm should
  #                be checked
  #
  # Returns:
  #   A list of parameters according to user input, default otherwise.
  if (!is.numeric(epsilon) || epsilon <= 0) {
    stop("value of 'epsilon' must be > 0")
  }
  return(list(epsilon=epsilon,
              trace=trace,
              deviance=deviance,
              convergence=convergence))
}

################################################################################
# Auxiliary functions: plots
################################################################################

# TODO
#plot.sgd.mse <- function(x) {
#  if (class(x) != "sgd") {
#    stop("'x' is not of type sgd")
#  }
#}

################################################################################
# Auxiliary functions: validity checks
################################################################################

sgd.model.control.valid <- function(model, model.control=list(...), ...) {
  # TODO documentation
  if (model == "glm") {
    control.family <- model.control$family
    control.intercept <- model.control$intercept
    # Check the validity of family.
    if (is.null(control.family)) {
      family <- gaussian()
    } else if (is.character(control.family)) {
      family <- get(family, mode="function", envir=parent.frame())()
    } else if (is.function(control.family)) {
      family <- family()
    } else if (is.null(control.family$family)) {
      print(family)
      stop("'family' not recognized")
    }
    # Check the validity of intercept.
    if (is.null(control.intercept)) {
      intercept <- TRUE
    } else if (!is.logical(control.intercept)) {
      stop("'intercept' not logical")
    }
    return(list(family=family, intercept=intercept))
  } else {
    stop("model not specified")
  }
}

sgd.sgd.control.valid <- function(method="implicit", lr.type="uni-dim",
                                  start=NULL, ...) {
  # TODO documentation
  # Check the validity of learning rate type.
  lr.types <- c("uni-dim", "uni-dim-eigen", "p-dim", "p-dim-weighted", "adagrad")
  if (is.numeric(lr.type)) {
    if (lr.type < 1 | lr.type > length(lr.types)) {
      stop("'lr.type' out of range")
    }
    lr.type <- lr.types[lr.type]
  } else if (is.character(lr.type)) {
    lr.type <- tolower(lr.type)
    if (!(lr.type %in% lr.types)) {
      stop("'lr.type' not recognized")
    }
  } else {
    stop("invalid 'lr.type'")
  }

  #Check the validity of start.
  if (!is.null(start) & !is.numeric(start)) {
    stop("'start' must be numeric")
  }
  # TODO where should we check if the dim(start) == dim(parameters)?

  # Check the validity of method.
  if (!is.character(method)) {
    stop("'method' must be a string")
  } else if (!(method %in% c("implicit", "asgd", "sgd"))) {
    stop("'method' not recognized")
  }
  return(list(method=method,
              lr.type=lr.type,
              start=start))
}

################################################################################
# Auxiliary functions: Miscellaneous
################################################################################

sgd.transfer.name <- function(link.name) {
  if(!is.character(link.name)) {
    stop("link name must be a string")
  }
  link.names <- c("identity", "log", "logit", "inverse")
  transfer.names <- c("identity", "exp", "logistic", "inverse")
  transfer.idx <- which(link.names == link.name)
  if (length(transfer.idx) == 0) {
    stop("no match link function founded!")
  }
  return(transfer.names[transfer.idx])
}
