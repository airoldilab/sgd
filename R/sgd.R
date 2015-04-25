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
  stop("class of x is not a formula, matrix, or function")
}

sgd.formula <- function(formula, model, data, model.control,
                        sgd.control=list(method="implicit", start=NULL,
                                         lr.type="uni-dim", ...),
                        ...) {
  # TODO
  # weights: how to weight using each data point; can be a parameter in sgd.control
  # subset: a subset of data points; can be a parameter in sgd.control
  # na.action: how to deal when data has NA; can be a parameter in sgd.control
  # model: logical value determining whether to output the X data frame
  # x,y: logical value determining whether to output the x and/or y
  # family: string determining which family in exponential family; can be a paramter in model.control (for GLMs)
  # offset: logical value determining whether to include intercept; can be a parameter in model.control
  # contrasts: a list for performing hypothesis testing on other sets of predictors; can be a paramter in sgd.control
  # Call method when the first argument is a formula
  # the call parameter to return
  call <- match.call()

  # 1. Safe check.
  if (missing(model)) {
    stop("model not specified")
  }

  # Get data from environment.
  if (missing(data)) {
    data <- environment(formula)
  }

  if (!missing(model.control)) {
    if (!is.list(model.control))  {
      stop("sgd.control is not a list")
    }

    # Set model.control according to user input and the default values.
    model.control <- do.call("sgd.model.valid", c(model.control, model=model))
  }

  # Set sgd.control according to user input and the default values.
  control <- do.call("sgd.sgd.control.valid", sgd.control)

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

  # 3. Fit!
  if (model == "glm") {
    fit <- sgd.fit.glm
  } else {
    print(model)
    stop("'model' not recognized")
  }
  out <- do.call("fit", c(list(x=X, y=Y), model.control, sgd.control))
  class(out) <- c(out$class, "sgd")
  return(out)
}

sgd.function <- function(x, fn.control=list(gr=NULL, lower=-Inf, upper=Inf),
                        sgd.control=list(method="implicit", start=NULL,
                                         lr.type="uni-dim", ...),
                        ...) {
  # TODO
  #
  # Args:
  #   x: loss function
  #   gr: gradient of loss function
  # TODO run_online_algorithm will not work on this as it relies on data
}

sgd.matrix <- function(x, y, model, model.control,
                        sgd.control=list(method="implicit", start=NULL,
                                         lr.type="uni-dim", ...),
                        ...) {
  # TODO
}

################################################################################
# Generic methods
################################################################################

print.sgd <- function() {}# TODO

################################################################################
# Auxiliary functions: model fitting
################################################################################

sgd.fit.glm <- function(x, y, weights=rep(1, nobs), start=NULL,
                     offset=rep(0, nobs), family=gaussian(), control=list(),
                     intercept=TRUE, method="implicit", lr.type, ...)  {
  control <- do.call("sgd.control.implicit", control)
  x <- as.matrix(x)
  xnames <- dimnames(x)[[2L]]
  ynames <- ifelse(is.matrix(y), rownames(y), names(y))
  conv <- FALSE
  nobs <- NROW(y)  # number of observations
  nvars <- ncol(x) # number of covariates
  EMPTY <- nvars == 0

  if (is.null(weights)) {
    weights <- rep.int(1, nobs)
  }
  if (is.null(offset)) {
    offset <- rep.int(0, nobs)
  }

  variance <- family$variance
  linkinv <- family$linkinv
  if (!is.function(variance) || !is.function(linkinv)) {
    stop("'family' argument seems not to be a valid family object",
         call.=FALSE)
  }
  dev.resids <- family$dev.resids
  aic <- family$aic
  mu.eta <- family$mu.eta

  unless.null <- function(x, if.null) ifelse(is.null(x), if.null, x)
  valideta <- unless.null(family$valideta, function(eta) TRUE)
  validmu <- unless.null(family$validmu, function(mu) TRUE)

  if (EMPTY) {
    eta <- rep.int(0, nobs) + offset
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
    if (!is.null(start) & length(start) != nvars) {
      stop(gettextf("length of 'start' should equal %d and correspond to initial coefs for %s",
                    nvars, paste(deparse(xnames), collapse=", ")), domain=NA)
    } else {
      start <- rep(0, nvars)
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
    experiment$deviance <- control$deviance
    experiment$trace <- control$trace
    experiment$convergence <- control$convergence
    experiment$epsilon <- control$epsilon
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
  n.ok <- nobs - sum(weights == 0)
  nulldf <- n.ok - as.integer(intercept)
  resdf <- n.ok - rank
  aic.model <- aic(y, 1, mu, weights, dev) + 2 * rank
  if (!EMPTY) {
    qr <- qr(x)
  }
  names(coef) <- xnames
  return(list(coefficients=coef,
       residuals=residuals,
       fitted.values=mu,
       R=if (!EMPTY) qr.R(qr),
       rank=rank,
       qr=if (!EMPTY) qr,
       family=family,
       linear.predictors=eta,
       deviance=dev,
       aic=aic.model,
       null.deviance=nulldev,
       iter=iter,
       weights=weights,
       df.residual=resdf,
       df.null=nulldf,
       y=y,
       estimates=if(!EMPTY) out$estimates,
       converged=if(control$convergence)
       converged))
  # TODO in C: deal with offset
  # TODO compare all results with glm
  # TODO unit test on all checks
  # TODO write start value
}

sgd.control.implicit <- function(epsilon=1e-08, trace=FALSE, deviance=FALSE,
                        convergence=FALSE) {
  # Set the control according to user input.
  if (!is.numeric(epsilon) || epsilon <= 0) {
    stop("value of 'epsilon' must be > 0")
  }
  list(epsilon=epsilon, trace=trace, deviance=deviance, convergence=convergence)
}


################################################################################
# Auxiliary functions: safe checking
################################################################################

sgd.sgd.control.valid <- function(method="implicit", start=NULL, lr.type="uni-dim", ...) {
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
  return(list(method=method, start=start, lr.type=lr.type))
}

sgd.model.valid <- function(model, temp=list(...), ...) {
  # TODO documentation
  family <- temp$family
  if (model == "glm") {
      # Check the validity of family.
    if (is.null("family")) family <- "gaussian"
    if (is.character(family)) {
      family <- get(family, mode="function", envir=parent.frame())
    }
    if (is.function(family)) {
      family <- family()
    }
    if (is.null(family$family)) {
      print(family)
      stop("'family' not recognized")
    }
    return(list(family=family))
  } else {
    stop("model not specified")
  }
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
