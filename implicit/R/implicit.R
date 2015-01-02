source('R/RcppExports.R')


#Set the control according to user input
implicit.control <- function(epsilon = 1e-08, maxit = 10000, trace = FALSE) 
{
  if (!is.numeric(epsilon) || epsilon <= 0) 
    stop("value of 'epsilon' must be > 0")
  if (!is.numeric(maxit) || maxit <= 0) 
    stop("maximum number of iterations must be > 0")
  list(epsilon = epsilon, maxit = maxit, trace = trace)
}

# A generic function to dispatch calls
implicit <- function(x, ...) UseMethod("implicit")

# if class(x) is formula, call implicit.formula

# Method to call when the first argument is a formula
implicit.formula <- function(formula, family = gaussian, data, weights, subset, 
                             na.action, start = NULL, etastart, mustart, offset, control = list(...), 
                             model = TRUE, method = "implicit", x = FALSE, y = TRUE, contrasts = NULL, 
                             ...){
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
  
  
  mf <- match.call(expand.dots = FALSE)

  #build dataframe according to the formula
  m <- match(c("formula", "data", "subset", "weights", "na.action", 
               "etastart", "mustart", "offset"), names(mf), 0L)
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
  mustart <- model.extract(mf, "mustart")
  etastart <- model.extract(mf, "etastart")
  
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
  fit <- implicit.fit(x=X, y=Y, family=family, method=method)
  return(fit)
  
  # model frame should be included as a component of the returned value
  if (model) 
    fit$model <- mf
  
  # calculate null.deviance: The deviance for the null model, comparable with deviance. 
  # The null model will include the offset, and an intercept if there is one in the model.
  if (length(offset) && attr(mt, "intercept") > 0L) {
    ######TODO: call fit for null model here
    fit2 <- list(converged = T, deviance = 0)
    if (!fit2$converged) 
      warning("fitting to calculate the null deviance did not converge -- increase 'maxit'?")
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
  link.names <- c("identity", "log", "logit")
  transfer.names <- c("identity", "exp", "logistic")
  match.indices <- match(link.names, link.name, 0L)
  if (sum(match.indices) == 0L) {
    stop("no match link function founded!")
  }
  transfer.idx = which(match.indices == 1L)
  transfer.names[transfer.idx]
}

implicit.fit <- function (x, y, weights = rep(1, nobs), start = NULL, etastart = NULL, 
                          mustart = NULL, offset = rep(0, nobs), family = gaussian(), 
                          control = list(), intercept = TRUE, method="implicit")  {
  x <- as.matrix(x)
  y <- as.matrix(y)
  dataset <- list(X=x, Y=y)
  
  experiment <- list()
  experiment$name = family$family
  experiment$transfer.name = implicit.transfer.name(family$link)
  experiment$niters = length(dataset$Y)
  experiment$lr = list(gamma0 = 1, alpha = 1, c = 2/3, scale = 1)
  experiment$p = dim(dataset$X)[2]
  
  out <- run_online_algorithm(dataset, experiment, method, F)
  out
}
