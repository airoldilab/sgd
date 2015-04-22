source("R/RcppExports.R")

sgd.control <- function(epsilon=1e-08, trace=FALSE, deviance=FALSE,
                        convergence=FALSE) {
  # Set the control according to user input.
  if (!is.numeric(epsilon) || epsilon <= 0) {
    stop("value of 'epsilon' must be > 0")
  }
  list(epsilon=epsilon, trace=trace, deviance=deviance, convergence=convergence)
}

sgd <- function(x, ...) UseMethod("sgd")
# If class(x) is formula, call sgd.formula.

sgd.formula <- function(formula, family=gaussian, data, weights, subset,
                        na.action, start=NULL, offset, control=list(...),
                        model=TRUE, method="implicit", x=FALSE, y=TRUE,
                        contrasts=NULL, lr.type="uni-dim", ...) {
  # Call method when the first argument is a formula
  # the call parameter to return
  call <- match.call()

  # Check the validity of family.
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

  # Get data from environment.
  if (missing(data)) {
    data <- environment(formula)
  }

  # Check the validity of method.
  if (!is.character(method)) {
    stop("'method' must be a string")
  } else if (!(method %in% c("implicit", "asgd", "sgd", "model.frame"))) {
    stop("'method' not recognized")
  }

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

  mf <- match.call(expand.dots=FALSE)

  # Build dataframe according to the formula.
  m <- match(c("formula", "data", "subset", "weights", "na.action",
               "offset"), names(mf), 0L)
  mf <- mf[c(1L, m)]
  mf$drop.unused.levels
  mf[[1L]] <- quote(stats::model.frame)
  mf <- eval(mf, parent.frame())

  # If method=="model.frame", return the dataframe without fitting.
  if (identical(method, "model.frame")) {
    return(mf)
  }

  mt <- attr(mf, "terms")
  # Set control according to user input and the default values.
  control <- do.call("sgd.control", control)

  Y <- model.response(mf, "any")
  if (length(dim(Y)) == 1L) {
    nm <- rownames(Y)
    dim(Y) <- NULL
    if (!is.null(nm)) {
      names(Y) <- nm
    }
  }

  if (!is.empty.model(mt)) {
    X <- model.matrix(mt, mf, contrasts)
  } else {
    X <- matrix(, NROW(Y), 0L)
  }

  # Check parameters for fitting.
  weights <- as.vector(model.weights(mf))
  if (!is.null(weights) && !is.numeric(weights)) {
    stop("'weights' must be a numeric vector")
  }
  if (!is.null(weights) && any(weights < 0)) {
    stop("negative weights not allowed")
  }
  offset <- as.vector(model.offset(mf))
  if (!is.null(offset)) {
    if (length(offset) != NROW(Y)) {
      stop(gettextf("number of offsets is %d should equal %d (number of observations)",
                    length(offset), NROW(Y)), domain=NA)
    }
  }

  fit <- sgd.fit(x=X, y=Y, weights=weights, start=start,
                 offset=offset, family=family, control=control,
                 intercept=attr(mt, "intercept") > 0L, method=method,
                 lr.type=lr.type)

  # Model frame should be included as a component of the returned value.
  if (model) {
    fit$model <- mf
  }

  # Calculate null.deviance: the deviance for the null model, comparable with
  # deviance.
  # The null model will include the offset, and an intercept if there is one in
  # the model.
  if (length(offset) && attr(mt, "intercept") > 0L) {
    fit2 <- sgd.fit(x=X[, "(Intercept)", drop=FALSE], y=Y, weights=weights,
                    offset=offset, family=family, control=control,
                    intercept=TRUE, method=method, lr.type=lr.type)
    fit$null.deviance <- fit2$deviance
  }

  # Include x and y in the returned value.
  if (x) {
    fit$x <- X
  }
  if (!y) {
    fit$y <- NULL
  }

  # The returned value should be the same as glm, so the returned object can be
  # used by all functions compatible with glm.
  fit <- c(fit, list(call=call, formula=formula, terms=mt,
                     data=data, offset=offset, control=control, method=method,
                     contrasts=attr(X, "contrasts"), xlevels=.getXlevels(mt, mf)))
  class(fit) <- c(fit$class, c("sgd", "glm", "lm"))
  return(fit)
}

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

sgd.fit <- function (x, y, weights=rep(1, nobs), start=NULL,
                     offset=rep(0, nobs), family=gaussian(), control=list(),
                     intercept=TRUE, method="implicit", lr.type, ...)  {
  control <- do.call("sgd.control", control)
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
