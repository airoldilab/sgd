source("R/RcppExports.R")

################################################################################
# Classes
################################################################################
sgd <- function(x, ...) UseMethod("sgd")
# If class(x) is formula, call sgd.formula.
# If class(x) is function, call sgd.function.
# If class(x) is matrix, call sgd.matrix.
# Otherwise, error.

################################################################################
# Methods
################################################################################

sgd.default <- function(x, ...) {
  stop("class of x is not a formula, function, or matrix")
}

sgd.formula <- function(formula, model, data,
                        model.control=list(),
                        sgd.control=list(...),
                        ...) {
  # Run stochastic gradient descent for model parameters.
  #
  # Args:
  #   formula:       formula specifying symbolic description of model
  #   model:         character in c("glm")
  #   data:          data frame for formula
  #   model.control: list of model-specific controls
  #     family:        "glm": string specifying which family in exponential
  #                    family
  #     intercept:     "glm": logical specifying whether to include intercept
  #   sgd.control:   list of optimization-specific controls
  #     method:        character in c("implicit", "sgd", "asgd")
  #     lr.type:       character in c("uni-dim", "uni-dim-eigen", "p-dim",
  #                                   "p-dim-weighted", "adagrad")
  #     start:         initial estimate
  #     weights:       how to weight using each data point
  #     offset:        how to offset the model
  #
  # Returns:
  #   sgd object
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

sgd.function <- function(x,
                        fn.control=list(),
                        sgd.control=list(...),
                        ...) {
  # Run stochastic gradient descent for model parameters.
  #
  # Args:
  #   x:             loss function
  #   fn.control:    list of function-specific controls
  #     gr:            gradient of loss function
  #     lower:         lower domain of loss function
  #     upper:         upper domain of loss function
  #   sgd.control:   list of optimization-specific controls
  #     method:        character in c("implicit", "sgd", "asgd")
  #     lr.type:       character in c("uni-dim", "uni-dim-eigen", "p-dim",
  #                                   "p-dim-weighted", "adagrad")
  #     start:         initial estimate
  #     weights:       how to weight using each data point
  #     offset:        how to offset the model
  #
  # Returns:
  #   sgd object
  # TODO run_online_algorithm will not work on this as it relies on data
  # sgd.fn.control.valid
  gr <- NULL
  lower <- -Inf
  upper <- Inf
}

sgd.matrix <- function(x, y, model,
                       model.control=list(),
                       sgd.control=list(...),
                       ...) {
  # Run stochastic gradient descent for model parameters.
  #
  # Args:
  #   formula:       formula specifying symbolic description of model
  #   model:         character in c("glm")
  #   data:          data frame for formula
  #   model.control: list of model-specific controls
  #     family:        "glm": string specifying which family in exponential
  #                    family
  #     intercept:     "glm": logical specifying whether to include intercept
  #   sgd.control:   list of optimization-specific controls
  #     method:        character in c("implicit", "sgd", "asgd")
  #     lr.type:       character in c("uni-dim", "uni-dim-eigen", "p-dim",
  #                                   "p-dim-weighted", "adagrad")
  #     start:         initial estimate
  #     weights:       how to weight using each data point
  #     offset:        how to offset the model
  #
  # Returns:
  #   sgd object
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

print.sgd <- function() {}# TODO

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
       y=y,
       estimates=if(!EMPTY) out$estimates,
       converged=if(implicit.control$convergence) converged))
  # TODO in C: deal with offset
  # TODO compare all results with glm
  # TODO unit test on all checks
  # TODO write start value
}

sgd.implicit.control <- function(epsilon=1e-08, trace=FALSE, deviance=FALSE,
                                 convergence=FALSE, ...) {
  # Set the control according to user input.
  if (!is.numeric(epsilon) || epsilon <= 0) {
    stop("value of 'epsilon' must be > 0")
  }
  return(list(epsilon=epsilon,
              trace=trace,
              deviance=deviance,
              convergence=convergence))
}


################################################################################
# Auxiliary functions: safe checking
################################################################################

sgd.model.control.valid <- function(model, model.control=list(...), ...) {
  # TODO documentation
  if (model == "glm") {
    family <- model.control$family
    intercept <- model.control$intercept
    # Check the validity of family.
    if (is.null("family")) {
      family <- gaussian()
    } else if (is.character(family)) {
      family <- get(family, mode="function", envir=parent.frame())()
    } else if (is.function(family)) {
      family <- family()
    } else if (is.null(family$family)) {
      print(family)
      stop("'family' not recognized")
    }
    # Check the validity of intercept.
    if (is.null(intercept)) {
      intercept <- TRUE
    } else if (!is.logical(intercept)) {
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
