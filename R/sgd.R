#' Stochastic gradient descent
#'
#' Run stochastic gradient descent in order to optimize the induced loss
#' function given a model and data.
#'
#' @param formula an object of class \code{"\link{formula}"} (or one that can be
#'   coerced to that class): a symbolic description of the model to be fitted.
#'   The details can be found in \code{"\link{glm}"}.
#' @param data an optional data frame, list or environment (or object coercible
#'   by \code{\link[base]{as.data.frame}} to a data frame) containing the
#'   variables in the model. If not found in data, the variables are taken from
#'   environment(formula), typically the environment from which glm is called.
#' @param model character specifying the model to be used: \code{"lm"} (linear
#'   model), \code{"glm"} (generalized linear model), \code{"cox"} (Cox
#'   proportional hazards model), \code{"gmm"} (generalized method of moments),
#'   \code{"m"} (M-estimation). See \sQuote{Details}.
#' @param model.control a list of parameters for controlling the model.
#'   \describe{
#'     \item{\code{family} (\code{"glm"})}{a description of the error distribution and
#'       link function to be used in the model. This can be a character string
#'       naming a family function, a family function or the result of a call to
#'       a family function. (See \code{\link[stats]{family}} for details of
#'       family functions.)}
#'     \item{\code{rank} (\code{"glm"})}{logical. Should the rank of the design matrix
#'       be checked?}
#'     \item{\code{fn} (\code{"gmm"})}{a function \eqn{g(\theta,x)} which returns a
#'       \eqn{k}-vector corresponding to the \eqn{k} moment conditions. It is a
#'       required argument if \code{gr} not specified.}
#'     \item{\code{gr} (\code{"gmm"})}{a function to return the gradient. If
#'       unspecified, a finite-difference approximation will be used.}
#'     \item{\code{nparams} (\code{"gmm"})}{number of model parameters. This is
#'       automatically determined for other models.}
#'     \item{\code{type} (\code{"gmm"})}{character specifying the generalized method of
#'       moments procedure: \code{"twostep"} (Hansen, 1982), \code{"iterative"}
#'       (Hansen et al., 1996). Defaults to \code{"iterative"}.}
#'     \item{\code{wmatrix} (\code{"gmm"})}{weighting matrix to be used in the loss
#'       function. Defaults to the identity matrix.}
#'     \item{\code{loss} (\code{"m"})}{character specifying the loss function to be
#'       used in the estimating equation. Default is the Huber loss.}
#'     \item{\code{lambda1}}{L1 regularization parameter. Default is 0.}
#'     \item{\code{lambda2}}{L2 regularization parameter. Default is 0.}
#'   }
#' @param sgd.control an optional list of parameters for controlling the estimation.
#'   \describe{
#'     \item{\code{method}}{character specifying the method to be used: \code{"sgd"},
#'       \code{"implicit"}, \code{"asgd"}, \code{"ai-sgd"}, \code{"momentum"},
#'       \code{"nesterov"}. Default is \code{"ai-sgd"}. See \sQuote{Details}.}
#'     \item{\code{lr}}{character specifying the learning rate to be used:
#'       \code{"one-dim"}, \code{"one-dim-eigen"}, \code{"d-dim"},
#'       \code{"adagrad"}, \code{"rmsprop"}. Default is \code{"one-dim"}.
#'       See \sQuote{Details}.}
#'     \item{\code{lr.control}}{vector of scalar hyperparameters one can
#'       set dependent on the learning rate. For hyperparameters aimed
#'       to be left as default, specify \code{NA} in the corresponding
#'       entries. See \sQuote{Details}.}
#'     \item{\code{start}}{starting values for the parameter estimates. Default is
#'       random initialization around zero.}
#'     \item{\code{size}}{number of SGD estimates to store for diagnostic purposes
#'       (distributed log-uniformly over total number of iterations)}
#'     \item{\code{reltol}}{relative convergence tolerance. The algorithm stops
#'       if it is unable to change the relative mean squared difference in the
#'       parameters by more than the amount. Default is \code{1e-05}.}
#'     \item{\code{npasses}}{the maximum number of passes over the data. Default
#'       is 3.}
#'     \item{\code{pass}}{logical. Should \code{tol} be ignored and run the
#'       algorithm for all of \code{npasses}?}
#'     \item{\code{shuffle}}{logical. Should the algorithm shuffle the data set
#'       including for each pass?}
#'     \item{\code{verbose}}{logical. Should the algorithm print progress?}
#'   }
#' @param \dots arguments to be used to form the default \code{sgd.control}
#'   arguments if it is not supplied directly.
#' @param fn a function \eqn{f(theta, x)} of parameters and data, which outputs
#'   a real number to be minimized.
#' @param gr a function to return the gradient. If it is \code{NULL}, a
#'   finite-difference approximation will be used.
#' @param x,y a design matrix and the respective vector of outcomes.
#'
#' @details
#' Models:
#' The Cox model assumes that the survival data is ordered when passed
#' in, i.e., such that the risk set of an observation i is all data points after
#' it.
#'
#' Methods:
#' \describe{
#'   \item{\code{sgd}}{stochastic gradient descent (Robbins and Monro, 1951)}
#'   \item{\code{implicit}}{implicit stochastic gradient descent (Toulis et al.,
#'     2014)}
#'   \item{\code{asgd}}{stochastic gradient with averaging (Polyak and Juditsky,
#'     1992)}
#'   \item{\code{ai-sgd}}{implicit stochastic gradient with averaging (Toulis et
#'     al., 2015)}
#'   \item{\code{momentum}}{"classical" momentum (Polyak, 1964)}
#'   \item{\code{nesterov}}{Nesterov's accelerated gradient (Nesterov, 1983)}
#' }
#'
#' Learning rates and hyperparameters:
#' \describe{
#'   \item{\code{one-dim}}{scalar value prescribed in Xu (2011) as
#'     \deqn{a_n = scale * gamma/(1 + alpha*gamma*n)^(-c)}
#'     where the defaults are
#'     \code{lr.control = (scale=1, gamma=1, alpha=1, c)}
#'     where \code{c} is \code{1} if implemented without averaging,
#'     \code{2/3} if with averaging}
#'   \item{\code{one-dim-eigen}}{diagonal matrix
#'     \code{lr.control = NULL}}
#'   \item{\code{d-dim}}{diagonal matrix
#'     \code{lr.control = (epsilon=1e-6)}}
#'   \item{\code{adagrad}}{diagonal matrix prescribed in Duchi et al. (2011) as
#'     \code{lr.control = (eta=1, epsilon=1e-6)}}
#'   \item{\code{rmsprop}}{diagonal matrix prescribed in Tieleman and Hinton
#'     (2012) as
#'     \code{lr.control = (eta=1, gamma=0.9, epsilon=1e-6)}}
#' }
#'
#' @return
#' An object of class \code{"sgd"}, which is a list containing the following
#' components:
#' \item{model}{name of the model}
#' \item{coefficients}{a named vector of coefficients}
#' \item{converged}{logical. Was the algorithm judged to have converged?}
#' \item{estimates}{estimates from algorithm stored at each iteration
#'     specified in \code{pos}}
#' \item{pos}{vector of indices specifying the iteration number each estimate
#'     was stored for}
#' \item{times}{vector of times in seconds it took to complete the number of
#'     iterations specified in \code{pos}}
#' \item{model.out}{a list of model-specific output attributes}
#'
#' @author Dustin Tran, Tian Lan, Panos Toulis, Ye Kuang, Edoardo Airoldi
#' @references
#' John Duchi, Elad Hazan, and Yoram Singer. Adaptive subgradient methods for
#' online learning and stochastic optimization. \emph{Journal of Machine
#' Learning Research}, 12:2121-2159, 2011.
#'
#' Yurii Nesterov. A method for solving a convex programming problem with
#' convergence rate \eqn{O(1/k^2)}. \emph{Soviet Mathematics Doklady},
#' 27(2):372-376, 1983.
#'
#' Boris T. Polyak. Some methods of speeding up the convergence of iteration
#' methods. \emph{USSR Computational Mathematics and Mathematical Physics},
#' 4(5):1–17, 1964.
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
#' @useDynLib sgd
#' @import MASS
#' @importFrom Rcpp evalCpp

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
  call <- match.call() # set call function to match on arguments
  # 1. Validity check.
  if (missing(data)) {
    data <- environment(formula)
  }

  # 2. Build X and Y according to the formula.
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
sgd.function <- function(fn, gr=NULL, x, y,
                         nparams,
                         sgd.control=list(...),
                         ...) {
  model <- "gmm"
  model.control <- list(model="gmm", fn=fn, gr=gr, d=ncol(x), nparams=nparams)
  return(sgd.matrix(x, y, model, model.control, sgd.control))
}

#' @export
#' @rdname sgd
sgd.matrix <- function(x, y, model,
                       model.control=list(),
                       sgd.control=list(...),
                       ...) {
  call <- match.call() # set call function to match on arguments
  if (missing(x)) {
    stop("'x' not specified")
  }
  if (missing(y)) {
    stop("'y' not specified")
  }
  if (missing(model)) {
    stop("'model' not specified")
  }
  if (!is.list(model.control)) {
    stop("'model.control' is not a list")
  }
  model.control <- do.call("valid_model_control",
                           c(model.control, model=model, d=ncol(x)))
  if (!is.list(sgd.control))  {
    stop("'sgd.control' is not a list")
  }
  sgd.control <- do.call("valid_sgd_control",
                         c(sgd.control, N=NROW(y), nparams=model.control$nparams))

  return(fit(x, y, model, model.control, sgd.control))
}

#' @export
#' @rdname sgd
# TODO y should be allowed to be a big matrix too; it should be any combination
# (x is a big matrix, y is, etc.)
sgd.big.matrix <- function(x, y, model,
                       model.control=list(),
                       sgd.control=list(...),
                       ...) {
  return(sgd.matrix(x, y, model, model.control, sgd.control))
}

################################################################################
# Helper functions
################################################################################

fit <- function(x, y, model,
                model.control,
                sgd.control) {
  #time_start <- proc.time()[3] # TODO timer only starts here
  # TODO
  if (model == "gmm") {
    if (sgd.control$method %in% c("implicit", "ai-sgd")) {
      stop("implicit methods not implemented yet")
    }
  }

  dataset <- list(X=x, Y=as.matrix(y))
  if ('big.matrix' %in% class(x)) {
    dataset$big <- TRUE
    dataset[["bigmat"]] <- x@address
  } else {
    dataset$big <- FALSE
    dataset[["bigmat"]] <- new("externalptr")
  }

  if (model %in% c("lm", "glm")) {
    model.control$transfer <- transfer_name(model.control$family$link)
    family <- model.control$family
    model.control$family <- family$family
  }

  if (sgd.control$verbose) {
    print("Completed pre-processing attributes...")
    print("Running C++ algorithm...")
  }
  out <- run(dataset, model.control, sgd.control)
  if (sgd.control$verbose) {
    print("Completed C++ algorithm...")
  }
  if (length(out) == 0) {
    stop("An error has occured, program stopped")
  }
  class(out) <- "sgd"
  if (model %in% c("lm", "glm")) {
    out$model.out$transfer <- model.control$transfer
    out$model.out$family <- family
  }
  out$pos <- as.vector(out$pos)
  #out$times <- as.vector(out$times) + (proc.time()[3] - time_start) # C++ time + R time
  out$times <- as.vector(out$times)
  return(out)
}

valid_model_control <- function(model, model.control=list(...), ...) {
  # Run validity check of arguments passed to model.control given model. It
  # passes defaults to those unspecified and converts to the correct type if
  # possible; otherwise it errors.
  # Check validity of regularization parameters.
  lambda1 <- model.control$lambda1
  if (is.null(lambda1)) {
    lambda1 <- 0
  } else if (!is.numeric(lambda1)) {
    stop("'lambda1' must be numeric")
  } else if (length(lambda1) != 1) {
    stop(gettextf("length of 'lambda1' should equal %d", 1), domain=NA)
  }
  lambda2 <- model.control$lambda2
  if (is.null(lambda2)) {
    lambda2 <- 0
  } else if (!is.numeric(lambda2)) {
    stop("'lambda2' must be numeric")
  } else if (length(lambda2) != 1) {
    stop(gettextf("length of 'lambda2' should equal %d", 1), domain=NA)
  }
  nparams <- model.control$d
  # Set family to gaussian for linear model.
  if (model == "lm") {
    model.control$family <- gaussian()
  }
  if (model %in% c("lm", "glm")) {
    control.family <- model.control$family
    control.rank <- model.control$rank
    control.trace <- model.control$trace
    control.deviance <- model.control$deviance
    # Check validity of family.
    if (is.null(control.family)) {
      control.family <- gaussian()
    } else if (is.character(control.family)) {
      control.family <- get(control.family, mode="function", envir=parent.frame())()
    } else if (is.function(control.family)) {
      control.family <- control.family()
    } else if (is.null(control.family$family)) {
      print(control.family)
      stop("'family' not recognized")
    }
    # Check validity of rank.
    if (is.null(control.rank)) {
      control.rank <- FALSE
    } else if (!is.logical(control.rank)) {
      stop ("'rank' not logical")
    }
    # Check validity of trace.
    if (is.null(control.trace)) {
      control.trace <- FALSE
    } else if (!is.logical(control.trace)) {
      stop ("'trace' not logical")
    }
    # Check validity of deviance.
    if (is.null(control.deviance)) {
      control.deviance <- FALSE
    } else if (!is.logical(control.deviance)) {
      stop ("'deviance' not logical")
    }
    return(list(
      name=model,
      family=control.family,
      rank=control.rank,
      trace=control.trace,
      deviance=control.deviance,
      nparams=nparams,
      lambda1=lambda1,
      lambda2=lambda2))
  } else if (model == "cox") {
    return(list(
      name=model,
      nparams=nparams,
      lambda1=lambda1,
      lambda2=lambda2))
  } else if (model == "gmm") {
    control.fn <- model.control$fn
    control.gr <- model.control$gr
    control.nparams <- model.control$nparams
    control.type <- model.control$type
    control.wmatrix <- model.control$wmatrix
    # Check validify of moment function and its gradient.
    if (is.null(control.fn) && is.null(control.gr)) {
      stop("either 'fn' or 'gr' must be specified")
    } else if (!is.null(control.fn) && !is.function(control.fn)) {
      stop("'fn' not a function")
    } else if (!is.null(control.gr) && !is.function(control.gr)) {
      stop("'gr' not a function")
    } else if (!is.null(control.fn) && is.null(control.gr)) {
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
    # Check validity of nparams.
    if (is.null(control.nparams)) {
      stop("'nparams' not specified")
    } else if (!is.numeric(control.nparams) ||
               control.nparams - as.integer(control.nparams) != 0 ||
               control.nparams < 1) {
      stop("'nparams' must be a positive integer")
    }
    # Check validity of GMM type.
    if (is.null(control.type)) {
      control.type <- "iterative"
    } else if (!is.character(control.type)) {
      stop("'type' must be a string")
    # TODO implement cuee
    } else if (!(control.type %in% c("twostep", "iterative", "cuee"))) {
      stop("'type' not recognized")
    }
    # Check validity of weighting matrix.
    if (is.null(control.wmatrix)) {
      # do nothing, since will not store large matrix in R but in C++
    } else if (!is.matrix(control.wmatrix)) {
      stop("'wmatrix' not a matrix")
    # TODO check if dimensions are same as moment conditions
    #} else if (identical(dim(control.wmatrix), c(k,k))) {
    }
    return(list(
      name=model,
      gr=control.gr,
      type=control.type,
      nparams=control.nparams,
      lambda1=lambda1,
      lambda2=lambda2))
  } else if (model == "m") {
    control.loss <-  model.control$loss
    if (is.null(control.loss)) {
      control.loss <- "huber"
    } else if (control.loss != "huber") {
      stop ("'loss' not available")
    }
    return(list(
      name=model,
      loss=control.loss,
      nparams=nparams,
      lambda1=lambda1,
      lambda2=lambda2))
  } else {
    stop("model not specified")
  }
}

valid_sgd_control <- function(method="ai-sgd", lr="one-dim",
                              lr.control=NULL,
                              start=rnorm(nparams, mean=0, sd=1e-5),
                              size=100,
                              reltol=1e-5, npasses=3, pass=F,
                              shuffle=F, verbose=F,
                              truth=NULL, check=F,
                              N, nparams, ...) {
  # The following are internal parameters that can be used but aren't written in
  # the documentation for succinctness:
  #   check: logical, specifying whether to check against \code{truth} for
  #          convergence instead of using reltol
  #   truth: true set of parameters
  # TODO size isn't the correct thing since reltol means you don't know when it
  # ends. user should specify how often to store the iterates (how many per
  # iteration)
  # Run validity check of arguments passed to sgd.control. It passes defaults to
  # those unspecified and converts to the correct type if possible; otherwise it
  # errors.
  # Check validity of method.
  if (!is.character(method)) {
    stop("'method' must be a string")
  } else if (!(method %in% c("sgd", "implicit", "asgd", "ai-sgd", "momentum",
                             "nesterov"))) {
    stop("'method' not recognized")
  }

  # Check validity of learning rate.
  lrs <- c("one-dim", "one-dim-eigen", "d-dim", "adagrad", "rmsprop")
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

  # Check validity of lr.control.
  if (!is.null(lr.control) && !is.numeric(lr.control)) {
    stop("'lr.control' must be numeric")
  } else if (lr == "one-dim") {
    if (method %in% c("asgd", "ai-sgd")) {
      c <- 2/3
    } else {
      c <- 1
    }
    defaults <- c(1, 1, 1, c)
    if (is.null(lr.control)) {
      lr.control <- defaults
    } else if (length(lr.control) != 4) {
      stop(gettextf("length of 'lr.control' should equal %d", 4), domain=NA)
    }
    missing <- which(is.na(lr.control))
    lr.control[missing] <- defaults[missing]
  } else if (lr == "one-dim-eigen") {
    if (is.null(lr.control)) {
      lr.control <- 0 # garbage number to store double in C++
    } else if (length(lr.control) != 0) {
      stop(gettextf("length of 'lr.control' should equal %d", 0), domain=NA)
    }
  } else if (lr == "d-dim") {
    defaults <- 1e-6
    if (is.null(lr.control)) {
      lr.control <- defaults
    } else if (length(lr.control) != 1) {
      stop(gettextf("length of 'lr.control' should equal %d", 1), domain=NA)
    }
    missing <- which(is.na(lr.control))
    lr.control[missing] <- defaults[missing]
  } else if (lr == "adagrad") {
    defaults <- c(1, 1e-6)
    if (is.null(lr.control)) {
      lr.control <- defaults
    } else if (length(lr.control) != 2) {
      stop(gettextf("length of 'lr.control' should equal %d", 2), domain=NA)
    }
    missing <- which(is.na(lr.control))
    lr.control[missing] <- defaults[missing]
  } else if (lr == "rmsprop") {
    defaults <- c(1, 0.9, 1e-6)
    if (is.null(lr.control)) {
      lr.control <- defaults
    } else if (length(lr.control) != 3) {
      stop(gettextf("length of 'lr.control' should equal %d", 3), domain=NA)
    }
    missing <- which(is.na(lr.control))
    lr.control[missing] <- defaults[missing]
  }

  # Check validity of start.
  if (!is.numeric(start)) {
    stop("'start' must be numeric")
  } else if (length(start) != nparams) {
    stop(gettextf("length of 'start' should equal %d", nparams), domain=NA)
  }

  # Check validity of size.
  if (!is.numeric(size) || size - as.integer(size) != 0 || size < 1) {
    stop("'size' must be positive integer")
  }

  # Check validity of reltol
  if (!is.numeric(reltol)) {
    stop("'reltol' must be numeric")
  } else if (length(reltol) != 1) {
    stop("'reltol' must be scalar")
  }

  # Check validity of npasses.
  if (!is.numeric(npasses) || npasses - as.integer(npasses) != 0 || npasses < 1) {
    stop("'npasses' must be positive integer")
  }

  # Check validity of pass.
  if (!is.logical(pass)) {
    stop("'pass' must be logical")
  }

  # Check validity of shuffle.
  if (!is.logical(shuffle)) {
    stop("'shuffle' must be logical")
  }

  # Check validity of verbose.
  if (!is.logical(verbose)) {
    stop("'verbose' must be logical")
  }

  # Check validity of additional arguments if the method is implicit.
  if (method %in% c("implicit", "ai-sgd")) {
    call <- match.call()
    implicit.control <- do.call("valid_implicit_control", list(...))
  } else {
    implicit.control <- NULL
  }

  # TODO they should be vectors in C++, not requiring conversion
  start <- as.matrix(start)
  if (check) {
    truth <- as.matrix(truth)
  }

  return(c(list(method=method,
                lr=lr,
                lr.control=lr.control,
                start=start,
                size=size,
                reltol=reltol,
                npasses=npasses,
                pass=pass,
                shuffle=shuffle,
                verbose=verbose,
                check=check,
                truth=truth,
                nparams=nparams),
           implicit.control))
}

valid_implicit_control <- function(delta=30L, ...) {
  # Maintain control parameters for running implicit SGD. Pass defaults
  # if unspecified.
  #
  # Args:
  #   delta: convergence criterion for the one-dimensional optimization
  if (!is.numeric(delta) || delta - as.integer(delta) != 0 || delta <= 0) {
    stop("value of 'delta' must be integer > 0")
  }
  return(list(delta=delta))
}

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
