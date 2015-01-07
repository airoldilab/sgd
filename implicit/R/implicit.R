source('R/RcppExports.R')

############################################
##TODO:Give warning if not converged?     ##
############################################
#Set the control according to user input
implicit.control <- function(epsilon = 1e-08, trace = FALSE, deviance = FALSE, convergence=FALSE) 
{
  if (!is.numeric(epsilon) || epsilon <= 0) 
    stop("value of 'epsilon' must be > 0")
  list(epsilon = epsilon, trace = trace, deviance = deviance, convergence=convergence)
}

# A generic function to dispatch calls
implicit <- function(x, ...) UseMethod("implicit")

# if class(x) is formula, call implicit.formula

# Method to call when the first argument is a formula
implicit.formula <- function(formula, family = gaussian, data, weights, subset, 
                             na.action, start = NULL, offset, control = list(...), 
                             model = TRUE, method = "implicit", x = FALSE, y = TRUE, contrasts = NULL, 
                             lr.type = "uni-dim", ...){
  #the call parameter to return
  call <- match.call()
  
  #check the validity of family, and make it a valid family object
  if (is.character(family)) 
    family <- get(family, mode = "function", envir = parent.frame())
  if (is.function(family)) 
    family <- family()
  if (is.null(family$family)) {
    print(family)
    stop("'family' not recognized")
  }
  
  #check the validity of method
  if (!is.character(method))
    stop("'method' must be a string")
  if (!(method %in% c('implicit', 'asgd', 'sgd', 'model.frame')))
    stop("'method' not recognized")
  
  #check the validity of learning rate type
  lr.types = c('uni-dim', 'px-dim')
  if (is.numeric(lr.type)) {
    if (lr.type < 1 | lr.type > 2) {
      stop("'lr.type' out of range")
    }
    lr.type = lr.types[lr.type]
  }
  else if (is.character(lr.type)) {
    if (!(lr.type %in% lr.types)) {
      stop("'lr.type' not recognized")
    }
  }
  else {
    stop("invalid 'lr.type'")
  }
  
  mf <- match.call(expand.dots = FALSE)

  #return(list(call=call, mf=mf))
  #build dataframe according to the formula
  m <- match(c("formula", "data", "subset", "weights", "na.action", 
               "offset"), names(mf), 0L)
  if (!is.character(method) && !is.function(method)) 
    stop("invalid 'method' argument")
  mf <- mf[c(1L, m)]
  mf$drop.unused.levels
  mf[[1L]] <- quote(stats::model.frame)
  mf <- eval(mf, parent.frame())
  
  # if method=='model.frame', return the dataframe without fitting
  if (identical(method, "model.frame")) 
    return(mf)
  
  mt <- attr(mf, "terms")
  #set control according to user input and the default values
  control <- do.call("implicit.control", control)
  
  Y <- model.response(mf, "any")
  if (length(dim(Y)) == 1L) {
    nm <- rownames(Y)
    dim(Y) <- NULL
    if (!is.null(nm)) 
      names(Y) <- nm
  }
  

  X <- if (!is.empty.model(mt)) 
    model.matrix(mt, mf, contrasts) else
    matrix(, NROW(Y), 0L)
  
  #check parameters for fitting
  weights <- as.vector(model.weights(mf))
  if (!is.null(weights) && !is.numeric(weights)) 
    stop("'weights' must be a numeric vector")
  if (!is.null(weights) && any(weights < 0)) 
    stop("negative weights not allowed")
  offset <- as.vector(model.offset(mf))
  if (!is.null(offset)) {
    if (length(offset) != NROW(Y)) 
      stop(gettextf("number of offsets is %d should equal %d (number of observations)", 
                    length(offset), NROW(Y)), domain = NA)
  }
  
  ###### TODO: plug in fit function
  # fit function should return 
  # list(coefficients = coef, residuals = residuals, fitted.values = mu, 
  # effects = if (!EMPTY) fit$effects, R = if (!EMPTY) Rmat, 
  # rank = rank, qr = if (!EMPTY) structure(fit[c("qr", "rank", 
  #                                               "qraux", "pivot", "tol")], class = "qr"), family = family, 
  # linear.predictors = eta, deviance = dev, aic = aic.model, 
  # null.deviance = nulldev, iter = iter, weights = wt, prior.weights = weights, 
  # df.residual = resdf, df.null = nulldf, y = y, converged = conv, 
  # boundary = boundary)
  
  fit <- implicit.fit(x = X, y = Y, weights = weights, start = start,
                      offset = offset, family = family, 
                      control = control, intercept = attr(mt, "intercept") > 0L,
                      method = method, lr.type=lr.type)
  
  # model frame should be included as a component of the returned value
  if (model) 
    fit$model <- mf
  
  # calculate null.deviance: The deviance for the null model, comparable with deviance. 
  # The null model will include the offset, and an intercept if there is one in the model.
  if (length(offset) && attr(mt, "intercept") > 0L) {
    ######TODO: call fit for null model here
    fit2 <- implicit.fit(x = X[, "(Intercept)", drop = FALSE], y = Y, weights = weights, 
                      offset = offset, family = family, control = control, 
                      intercept = TRUE)
    fit$null.deviance <- fit2$deviance
  }
  
  # include x and y in the returned value
  if (x) 
    fit$x <- X
  if (!y) 
    fit$y <- NULL
  
  # The returned value should be the same as glm, so the resturned object can be used
  # by all functions compatible with glm
  fit <- c(fit, list(call = call, formula = formula, terms = mt, 
                     data = data, offset = offset, control = control, method = method, 
                     contrasts = attr(X, "contrasts"), xlevels = .getXlevels(mt, mf)))
  class(fit) <- c(fit$class, c("implicit", "glm", "lm"))
  fit
}

implicit.transfer.name <- function(link.name) {
  if(!is.character(link.name)) {
    stop("link name must be a string")
  }
  link.names <- c("identity", "log", "logit", "inverse")
  transfer.names <- c("identity", "exp", "logistic", "inverse")
  match.indices <- match(link.names, link.name, 0L)
  if (sum(match.indices) == 0L) {
    stop("no match link function founded!")
  }
  transfer.idx = which(match.indices == 1L)
  transfer.names[transfer.idx]
}

implicit.fit <- function (x, y, weights = rep(1, nobs), start = NULL,
                          offset = rep(0, nobs), family = gaussian(), 
                          control = list(), intercept = TRUE, method="implicit", lr.type)  {
  control <- do.call("implicit.control", control)
  x <- as.matrix(x)
  xnames <- dimnames(x)[[2L]]
  ynames <- if (is.matrix(y)) 
    rownames(y) else
    names(y)
  conv <- FALSE
  nobs <- NROW(y)  # number of observations
  nvars <- ncol(x) # number of covariates
  EMPTY <- nvars == 0
  
  if (is.null(weights)) 
    weights <- rep.int(1, nobs)
  if (is.null(offset)) 
    offset <- rep.int(0, nobs)
  
  variance <- family$variance
  linkinv <- family$linkinv
  if (!is.function(variance) || !is.function(linkinv)) 
    stop("'family' argument seems not to be a valid family object", 
         call. = FALSE)
  dev.resids <- family$dev.resids
  aic <- family$aic
  mu.eta <- family$mu.eta
  
  unless.null <- function(x, if.null) 
    if (is.null(x)) 
    if.null else x
  valideta <- unless.null(family$valideta, function(eta) TRUE)
  validmu <- unless.null(family$validmu, function(mu) TRUE)
  
  if (EMPTY) {
    eta <- rep.int(0, nobs) + offset
    if (!valideta(eta)) 
      stop("invalid linear predictor values in empty model", 
           call. = FALSE)
    mu <- linkinv(eta)
    if (!validmu(mu)) 
      stop("invalid fitted means in empty model", call. = FALSE)
    dev <- sum(dev.resids(y, mu, weights))
    w <- ((weights * mu.eta(eta)^2)/variance(mu))^0.5
    residuals <- (y - mu)/mu.eta(eta)
    good <- rep_len(TRUE, length(residuals))
    boundary <- conv <- TRUE
    coef <- numeric()
    iter <- 0L
    rank <- 0L
  } else
  {
    #set the initial value for theta
    start <- if (!is.null(start))
      if (length(start) != nvars) 
        stop(gettextf("length of 'start' should equal %d and correspond to initial coefs for %s", 
                      nvars, paste(deparse(xnames), collapse = ", ")), domain = NA) else
      start else {
        rep(0, nvars)
      }
    eta = sum(x[1, ] * start)+offset[1]
    if (!valideta(eta))
      stop("cannot find valid starting values: please specify some", call. = FALSE)
    y <- as.matrix(y)
    
    #select x, y with weights>0, adjust for offsets
    good <- weights > 0
    dataset <- list(X=as.matrix(x[good, ]), Y=as.matrix(y[good]))
    experiment <- list()
    experiment$name = family$family
    experiment$transfer.name = implicit.transfer.name(family$link)
    experiment$niters = length(dataset$Y)
    experiment$lr.type = lr.type
    experiment$p = dim(dataset$X)[2]
    experiment$weights = as.matrix(weights[good])
    experiment$start = as.matrix(start)
    experiment$deviance = control$deviance
    experiment$trace = control$trace
    experiment$convergence = control$convergence
    experiment$epsilon = control$epsilon
    experiment$offset = as.matrix(offset)
    out <- run_online_algorithm(dataset, experiment, method, F)
    
    if (length(out) == 0) {
      stop("An error has occured, program stopped. ")
    }
    mu = as.numeric(out$mu)
    eta = as.numeric(out$eta)
    coef = as.numeric(out$coefficients)
    dev = out$deviance
    residuals = as.numeric((y - mu)/mu.eta(eta))
    iter = experiment$p
    rank = out$rank
  }
  names(residuals) <- ynames
  names(mu) <- ynames
  names(eta) <- ynames
  names(weights) <- ynames
  names(y) <- ynames
  wtdmu <- if (intercept) 
    sum(weights * y)/sum(weights) else
    linkinv(offset)
  nulldev <- sum(dev.resids(y, wtdmu, weights))
  n.ok <- nobs - sum(weights == 0)
  nulldf <- n.ok - as.integer(intercept)
  resdf <- n.ok - rank
  aic.model <- aic(y, 1, mu, weights, dev) + 2 * rank
  if (!EMPTY)
    qr = qr(x)
  names(coef) <- xnames
  list(coefficients = coef, residuals = residuals, fitted.values = mu, 
       R = if (!EMPTY) qr.R(qr), 
       rank = rank, qr = if (!EMPTY) qr, family = family, 
       linear.predictors = eta, deviance = dev, aic = aic.model, 
       null.deviance = nulldev, iter = iter, weights = weights, 
       df.residual = resdf, df.null = nulldf, y = y, 
       estimates = if(!EMPTY) out$estimates)
  ######TODO in C: deal with offset
  ######TODO compare all results with glm
  ######TODO unit test on all checks
  ######TODO write start value
}
