#' Stochastic gradient descent
#'
#' Run stochastic gradient descent on the underlying loss function for a given
#' model and data, or a user-specified loss function.
#'
#' @param formula an object of class \code{"\link{formula}"} (or one that can be
#'   coerced to that class): a symbolic description of the model to be fitted.
#'   The details can be found in \code{"\link{glm}"}.
#' @param data an optional data frame, list or environment (or object coercible
#'   by \code{\link[base]{as.data.frame}} to a data frame) containing the
#'   variables in the model. If not found in data, the variables are taken from
#'   environment(formula), typically the environment from which glm is called.
#' @param model character specifying the model to be used: \code{"lm"} (linear
#'   model), \code{"glm"} (generalized linear model), \code{"ee"} (estimating
#'   equation).
#' @param model.control a list of parameters for controlling the model.
#'   \itemize{
#'     \item family (\code{"glm"}): a description of the error distribution and
#'       link function to be used in the model. This can be a character string
#'       naming a family function, a family function or the result of a call to
#'       a family function.  (See \code{\link[stats]{family}} for details of
#'       family functions.)
#'     \item intercept (\code{"lm"}, \code{"glm"}): logical. Should an intercept
#'       be included in the \emph{null} model?
#'     \item rank logical. Should the rank of the design matrix be checked?
#'     \item fn (\code{"ee"}): moment function, which is a required argument if
#'       \code{gr} not specified
#'     \item gr (\code{"ee"}): gradient of the moment function, which if not
#'       passed in defaults to taking the numerical gradient of \code{fn}
#'   }
#' @param sgd.control a list of parameters for controlling the estimation
#'   \itemize{
#'     \item method: character specifying the method to be used: \code{"sgd"},
#'     \code{"implicit"}, \code{"asgd"}, \code{"ai-sgd"}. Default is
#'     \code{"implicit"}.  See \sQuote{Details}.
#'     \item lr: character specifying the learning rate to be used:
#'       \code{"one-dim"}, \code{"one-dim-eigen"}, \code{"d-dim"},
#'       \code{"adagrad"}. Default is \code{"one-dim"}.
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
#' @param x for \code{sgd.function}, x is a function to minimize; for
#' \code{sgd.matrix}, x is a design matrix.
#' @param y for {sgd.matrix}, y is a vector of observations, with length equal
#' to the number of rows in x.
#' @param fn.control for \code{sgd.function}, it is a list of controls for the
#' function.
#'
#' @details
#' Methods:
#' \itemize{
#'   \item \code{sgd}: stochastic gradient descent (Robbins and Monro, 1951)
#'   \item \code{implicit}: implicit stochastic gradient descent (Toulis et al.,
#'     2014)
#'   \item \code{asgd} stochastic gradient with averaging (Polyak and Juditsky,
#'     1992)
#'   \item \code{ai-sgd} implicit stochastic gradient with averaging (Toulis et
#'     al., 2015)
#' }
#'
#' Learning rates:
#' \itemize{
#'   \item \code{one-dim}: scalar value prescribed in Xu (2011) as
#'     \code{a_n = scale * gamma/(1 + alpha*gamma*n)^(-c)}
#'     where the defaults are \code{scale=1}, \code{gamma=1}, \code{alpha} the
#'     minimum eigenvalue of the covariance of the data's design matrix, and
#'     \code{c} is \code{1} if implemented without averaging, \code{2/3} if with
#'     averaging
#'   \item \code{one-dim-eigen}: diagonal matrix
#'     \code{}
#'   \item \code{d-dim}: diagonal matrix
#'     \code{}
#'   \item \code{adagrad}: diagonal matrix prescribed in Duchi et al. (2011) as
#'     \code{}
#' }
#'
#' @return
#' An object of class \code{"sgd"}, which is a list containing at least the
#' following components:
#'
#' \code{coefficients}
#' a named vector of coefficients
#'
#' \code{converged}
#' logical. Was the algorithm judged to have converged?
#'
#' \code{model.out}
#' a list of model-specific output attributes
#'
#' @author Dustin Tran, Tian Lan, Panos Toulis, Ye Kuang, Edoardo Airoldi
#' @references
#' John Duchi, Elad Hazan, and Yoram Singer. Adaptive subgradient methods for
#' online learning and stochastic optimization. \emph{Journal of Machine
#' Learning Research}, 12:2121-2159, 2011.
#'
#' Boris T. Polyak and Anatoli B. Juditsky. Acceleration of stochastic
#' approximation by averaging. \emph{SIAM Journal on Control and Optimization},
#' 30(4):838-855, 1992.
#'
#' Herbert Robbins and Sutton Monro. A stochastic approximation method.
#' \emph{The Annals of Mathematical Statistics}, pp. 400-407, 1951.
#'
#' Panos Toulis, Jason Rennie, and Edoardo M. Airoldi, "Statistical analysis of
#' stochastic gradient methods for generalized linear models", In
#' \emph{Proceedings of the 31st International Conference on Machine Learning},
#' 2014.
#'
#' Panos Toulis, Dustin Tran, and Edoardo M. Airoldi, "Stability and optimality
#' in stochastic gradient descent", arXiv preprint arXiv:1505.02417, 2015.
#'
#' Wei Xu. Towards optimal one pass large scale learning with averaged
#' stochastic gradient descent. arXiv preprint arXiv:1107.2490, 2011.
#'
#' @examples
#' ## Dobson (1990, p.93): Randomized Controlled Trial
#' counts <- c(18, 17, 15, 20, 10, 20, 25, 13, 12)
#' outcome <- gl(3, 1, 9)
#' treatment <- gl(3, 3)
#' print(d.AD <- data.frame(treatment, outcome, counts))
#' sgd.D93 <- sgd(counts ~ outcome + treatment, model="glm",
#'                model.control=list(family = poisson()))
#' sgd.D93
#'
#' ## Venables & Ripley (2002, p.189): an example with offsets
#' utils::data(anorexia, package="MASS")
#'
#' anorex.1 <- sgd(Postwt ~ Prewt + Treat + offset(Prewt),
#'                 data=anorexia, model="lm")
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
  # Set call function to match on arguments
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
  # TODO run_online_algorithm will not work as it relies on data
  # default args for fn.control
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
  # Set call function to match on arguments
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
  }
  model.control <- do.call("valid_model_control", c(model.control, model=model))
  if (!is.list(sgd.control))  {
    stop("'sgd.control' is not a list")
  }
  sgd.control <- do.call("valid_sgd_control", c(sgd.control, N=NROW(y),
    d=ncol(x)))

  # 2. Fit!
  if (model %in% c("lm", "glm")) {
    fit <- fit_glm
  } else if (model == "ee") {
    fit <- fit_ee
  } else {
    print(model)
    stop("'model' not recognized")
  }
  out <- fit(x, y, model.control, sgd.control)
  if (nrow(x) > 200){
    samples = sample(nrow(x), 200,replace = F)
  }
  else {
    samples = 1:nrow(x)
  }
  sample.x <- x[samples, ]
  sample.y <- y[samples]
  classes <- c(class(out), "sgd")
  out <- c(out, list(sample.x=sample.x, sample.y=sample.y))
  class(out) <- classes
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
#' @export
plot.sgd <- function(x, type="mse", ...) {
 # An all-encompassing visualization routine.
 #
 # Args:
 #   x:    sgd object
 #   type: character in c("")
 #
 # Returns:
 #   A plot.
 if (type == "mse") {
   plot <- plot.sgd.mse
 } else {
   print(type)
   stop("'type' not recognized")
 }
 return(plot(x))
}

################################################################################
# Auxiliary functions: model fitting
################################################################################

fit_glm <- function(x, y,
                    model.control,
                    sgd.control) {
  xnames <- dimnames(x)[[2L]]
  if (is.matrix(y)) {
    ynames <- rownames(y)
  } else {
    ynames <- names(y)
  }
  N <- NROW(y) # number of observations
  d <- ncol(x) # number of features

  # sgd.control arguments
  method <- sgd.control$method
  lr <- sgd.control$lr
  start <- sgd.control$start
  weights <- sgd.control$weight
  offset <- sgd.control$offset

  # model.control arguments
  family <- model.control$family
  intercept <- model.control$intercept

  variance <- family$variance
  linkinv <- family$linkinv
  if (!is.function(variance) || !is.function(linkinv)) {
    stop("'family' argument seems not to be a valid family object",
         call.=FALSE)
  }
  dev.resids <- family$dev.resids
  mu.eta <- family$mu.eta

  unless.null <- function(x, if.null) {
    if (is.null(x)) {
      return(if.null)
    } else {
      return(x)
    }
  }
  valideta <- unless.null(family$valideta, function(eta) TRUE)
  validmu <- unless.null(family$validmu, function(mu) TRUE)

  EMPTY <- d == 0
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
    experiment$niters <- length(dataset$Y)
    experiment$d <- dim(dataset$X)[2]
    experiment$lr <- lr
    experiment$start <- as.matrix(start)
    experiment$weights <- as.matrix(weights[good])
    experiment$offset <- as.matrix(offset[good]) # TODO not implemented
    experiment$epsilon <- sgd.control$epsilon
    experiment$trace <- sgd.control$trace
    experiment$deviance <- sgd.control$deviance
    experiment$convergence <- sgd.control$convergence
    experiment$model.attrs <- list()
    experiment$model.attrs$transfer.name <- transfer_name(family$link)
    experiment$model.attrs$rank <- model.control$rank

    out <- run_online_algorithm(dataset, experiment, method, verbose=F)
    if (length(out) == 0) {
      stop("An error has occured, program stopped")
    }
    temp.mu <- as.numeric(out$model.out$mu)
    mu <- rep(0, length(good))
    mu[good] <- temp.mu
    mu[!good] <- NA
    temp.eta <- as.numeric(out$model.out$eta)
    eta <- rep(0, length(good))
    eta[good] <- temp.eta
    eta[!good] <- NA
    coef <- as.numeric(out$coefficients)
    dev <- out$model.out$deviance
    residuals <- as.numeric((y - mu)/mu.eta(eta))
    iter <- experiment$p
    rank <- out$model.out$rank
    converged <- out$converged
  }
  names(residuals) <- ynames
  names(mu) <- ynames
  names(eta) <- ynames
  names(weights) <- ynames
  names(y) <- ynames
  if (intercept == TRUE) {
    wtdmu <- sum(weights * y)/sum(weights)
  } else {
    wtdmu <- linkinv(offset)
  }
  nulldev <- sum(dev.resids(y, wtdmu, weights))
  n.ok <- N - sum(weights == 0)
  nulldf <- n.ok - as.integer(intercept)
  resdf <- n.ok - rank
  names(coef) <- xnames
  result <- list(
    coefficients=coef,
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
    converged=if(sgd.control$convergence) converged,
    estimates=out$estimates,
    pos=out$pos,
    aic=0)
  class(result) <- c(class(result), "glm")
  return(result)
}

fit_ee <- function(x, y,
                   model.control,
                   sgd.control) {
  # TODO
  if (sgd.control$method %in% c("implicit", "ai-sgd")) {
    stop("implicit methods not implemented yet")
  }

  xnames <- dimnames(x)[[2L]]
  if (is.matrix(y)) {
    ynames <- rownames(y)
  } else {
    ynames <- names(y)
  }
  N <- NROW(y) # number of observations
  d <- ncol(x) # number of features

  EMPTY <- d == 0
  if (EMPTY) {
    # TODO
  } else {
    dataset <- list(X=x, Y=y)
    experiment <- list()
    experiment$name <- "ee"
    experiment$niters <- N
    experiment$d <- d
    experiment$lr <- sgd.control$lr
    experiment$start <- as.matrix(sgd.control$start)
    experiment$weights <- sgd.control$weights # TODO not implemented
    experiment$offset <- sgd.control$offset # TODO not implemented
    experiment$epsilon <- sgd.control$epsilon
    experiment$trace <- sgd.control$trace
    experiment$deviance <- sgd.control$deviance
    experiment$convergence <- sgd.control$convergence
    experiment$model.attrs <- list()
    experiment$model.attrs$fn <- model.control$fn
    experiment$model.attrs$gr <- model.control$gr

    out <- run_online_algorithm(dataset, experiment, sgd.control$method, verbose=F)
    if (length(out) == 0) {
      stop("An error has occured, program stopped")
    }
  }

  return(list(
    coefficients=out$coef,
    converged=out$converged,
    estimates=out$estimates
    ))
}

################################################################################
# Auxiliary functions: plots
################################################################################

sgd.mse.glm <- function(x){
  eta <- x$sample.x %*% x$estimates
  mu <- x$family$linkinv(eta)
  mse <- colMeans((mu - x$sample.y)^2)
  return(mse)
}

plot.sgd.mse <- function(x){
  if (any(class(x) %in% "glm")){
    get.mse <- sgd.mse.glm
  }
  else{
    stop("Model not recognized! ")
  }
  mse <- get.mse(x)
  dat <- data.frame(mse=mse, pos=x$pos[1, ])
  dat <- dat[!duplicated(dat$pos), ]
  pos <- 0
  p <- ggplot2::ggplot(dat, ggplot2::aes(x=pos, y=mse)) + ggplot2::geom_line() +
    ggplot2::theme_bw() + ggplot2::theme(panel.border = ggplot2::element_blank(), panel.grid.major = ggplot2::element_blank(),
                       panel.grid.minor = ggplot2::element_blank(), axis.line = ggplot2::element_line(colour = "black")) +
    ggplot2::labs(title = "Mean Squared Error", x = "Iteration", y = "MSE")
  return(p)
}

################################################################################
# Auxiliary functions: validity checks
################################################################################

valid_model_control <- function(model, model.control=list(...), ...) {
  # Run validity check of arguments passed to model.control given model. It
  # passes defaults to those unspecified and converts to the correct type if
  # possible; otherwise it errors.
  if (model == "lm") {
    control.intercept <- model.control$intercept
    control.rank <- model.control$rank
    # Check the validity of intercept.
    if (is.null(control.intercept)) {
      control.intercept <- TRUE
    } else if (!is.logical(control.intercept)) {
      stop("'intercept' not logical")
    }
    # Check the validity of rank.
    if (is.null(control.rank)){
      control.rank <- FALSE
    } else if (!is.logical(control.intercept)) {
      stop ("'rank' not logical")
    }
    return(list(family=gaussian(), intercept=control.intercept, rank=control.rank))
  } else if (model == "glm") {
    control.family <- model.control$family
    control.intercept <- model.control$intercept
    control.rank <- model.control$rank
    # Check the validity of family.
    if (is.null(control.family)) {
      control.family <- gaussian()
    } else if (is.character(control.family)) {
      control.family <- get(family, mode="function", envir=parent.frame())()
    } else if (is.function(control.family)) {
      control.family <- control.family()
    } else if (is.null(control.family$family)) {
      print(control.family)
      stop("'family' not recognized")
    }
    # Check the validity of intercept.
    if (is.null(control.intercept)) {
      control.intercept <- TRUE
    } else if (!is.logical(control.intercept)) {
      stop("'intercept' not logical")
    }
    # Check the validity of rank.
    if (is.null(control.rank)){
      control.rank <- FALSE
    } else if (!is.logical(control.intercept)) {
      stop ("'rank' not logical")
    }
    return(list(family=control.family, intercept=control.intercept, rank=control.rank))
  } else if (model == "ee") {
    # for now do iterative procedure
    control.fn <- model.control$fn
    control.gr <- model.control$gr
    # Check the validify of moment function and its gradient.
    if (is.null(control.fn)) {
      if (is.null(control.gr)) {
        stop("either 'fn' or 'gr' must be specified")
      } else if (!is.function(control.gr)) {
        stop("'gr' not a function")
      }
    } else if (!is.function(control.fn)) {
      stop("'fn' not a function")
    } else if (is.null(control.gr)) {
      # Default to numerical gradient via central differences.
      #library(numDeriv)
      # TODO probably does not work
      control.gr <- function(x, fn=control.fn) {
        d <- length(x)
        h <- 1e-5
        out <- rep(0, d)
        for (i in 1:d) {
          ei <- c(rep(0, i-1), h, rep(0, d-i))
          out[i] <- (fn(x + ei) - fn(x - ei)) / (2*h)
        }
        return(out)
      }
    }
    return(list(fn=control.fn, gr=control.gr))
  } else {
    stop("model not specified")
  }
}

valid_sgd_control <- function(method="implicit", lr="one-dim",
                              start=NULL, weights=NULL,
                              offset=NULL, N, d, ...) {
  # Run validity check of arguments passed to sgd.control. It passes defaults to
  # those unspecified and converts to the correct type if possible; otherwise it
  # errors.
  # Check validity of method.
  if (!is.character(method)) {
    stop("'method' must be a string")
  } else if (!(method %in% c("sgd", "implicit", "asgd", "ai-sgd"))) {
    stop("'method' not recognized")
  }

  # Check validity of learning rate.
  lrs <- c("one-dim", "one-dim-eigen", "d-dim", "adagrad")
  if (is.numeric(lr)) {
    if (lr < 1 | lr > length(lrs)) {
      stop("'lr' out of range")
    }
    lr <- lrs[lr]
  } else if (is.character(lr)) {
    lr <- tolower(lr)
    if (!(lr %in% lrs)) {
      stop("'lr' not recognized")
    }
  } else {
    stop("invalid 'lr'")
  }

  # Check validity of start.
  if (is.null(start)) {
    start <- rep(0, d)
  } else if (!is.numeric(start)) {
    stop("'start' must be numeric")
  } else if (length(start) != d) {
    stop(gettextf("length of 'start' should equal %d", d), domain=NA)
  }

  # Check validity of weights.
  if (is.null(weights)) {
    weights <- rep.int(1, N)
  } else if (!is.numeric(weights)) {
    stop("'weights' must be numeric")
  } else if (length(weights) != N) {
    stop(gettextf("length of 'weights' should equal %d", N), domain=NA)
  }

  # Check validity of offset.
  if (is.null(offset)) {
    offset <- rep.int(0, N)
  } else if (!is.numeric(offset)) {
    stop("'offset' must be numeric")
  } else if (length(offset) != N) {
    stop(gettextf("length of 'offset' should equal %d", N), domain=NA)
  }

  # Check validity of additional arguments if the method is implicit.
  if (method %in% c("implicit", "ai-sgd")) {
    call <- match.call()
    implicit.control <- do.call("valid_implicit_control", list(...))
  }

  return(c(list(method=method,
                lr=lr,
                start=start,
                weights=weights,
                offset=offset),
           implicit.control))
}

valid_implicit_control <- function(epsilon=1e-08, trace=FALSE, deviance=FALSE,
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
# Auxiliary functions: Miscellaneous
################################################################################

transfer_name <- function(link.name) {
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
