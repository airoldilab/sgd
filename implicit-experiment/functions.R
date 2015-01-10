library(mvtnorm)
library(implicit)
empirical.variance <- function(model, method, nreps, niters){
  
}

sample.x.normal <- function(niters, np, mean=rep(0,np), A){
  if (length(mean) != np)
    stop('dimension of mean is wrong')
  sample.covariance.matrix <- function(p) {
    # Samples a low-rank covariance matrix.
    u1 = 0.5 * seq(-1, 1, length.out=p)
    u2 = seq(0.2, 1, length.out=p)
    C = matrix(0, nrow=p, ncol=p)
    diag(C) <- u2
    V =  (C + u1 %*% t(u1))
    V
  }
  if (missing(A))
    A = sample.covariance.matrix(np)
  X = rmvnorm(niters, mean, sigma=A)
  X
}

sample.theta.const <- function(np, theta){
  theta = matrix(theta, ncol=1)
  theta
}

sample.y.normal <- function(mu, sd=1){
  y = rnorm(length(mu),mu, sd)
  y = matrix(y, ncol=1)
  y
}


sample.x.poisson <- function(niters, np, lambda=rep(10,np)){
  if (length(lambda) != np)
    stop('dimension of lambda is wrong')
  lambda = rep(lambda, niters)
  X = rpois(length(lambda), lambda)
  X = matrix(X, ncol=np, nrow=niters, byrow = T)
  X
}

sample.y.poisson <- function(mu){
  y = rpois(length(mu), mu)
  y = matrix(y, ncol=1)
  y
}


#model: the name of the model
#       Or a list of functions: sample.x(niters, np, ...), 
#                               sample.theta(np, ...) 
#                               sample.y(mu, ...)
#                               family object
#method: a string or a list of strings
#learning.rate: a string or a list of strings
#np: number of parameters
#nreps: repetitions in a experiment
#niters: number of iterations in one experiment
#...: x.control: a list of arguments to be passed to sample.x
#     theta.control: a list of arguments to be passed to sample.theta
#     y.control: a list of arguments to be passed to sample.x
empirical.variance <- function(model, method, learning.rate, np, nreps, niters, plot=F, ...){
  #determine how to sample x, y, z and the transfer function
  sample.x = NULL
  sample.theta = NULL
  sample.y = NULL
  x.control = NULL
  theta.control = NULL
  y.control = NULL
  control = list(...)
  family = NULL
  if ('x.control' %in% names(control)){
    x.control = control$x.control
  }
  if ('theta.control' %in% names(control)){
    theta.control = control$theta.control
  }
  if ('y.control' %in% names(control)){
    y.control = control$y.control
  }
  if (is.character(model)){
    if (!(model %in% c('normal', 'poisson')))
      stop('model not supported')
    if (model == 'normal'){
      sample.x = sample.x.normal
      sample.theta = sample.theta.const
      sample.y = sample.y.normal
      if (is.null(theta.control)){
        theta.control = list(theta = rep(1,np))
      }
      family = gaussian()
    }
    if (model == 'poisson'){
      sample.x = sample.x.poisson
      sample.theta = sample.theta.const
      sample.y = sample.y.poisson
      if (is.null(theta.control)){
        theta.control = list(theta = rep(1,np))
      }
      family = poisson()
    }
  } else{
    sample.x = model$sample.x
    sample.theta = model$sample.theta
    sample.y = model$sample.y
    family = model$family
  }
  #run the experiment
  #estimates[['implicit']][['uni-dim']][3] get the estimates of repetition 3
  estimates = list() 
  for (m in method){
    for (l in learning.rate){
      estimates[[m]][[l]] = array(dim=c(nreps, np, niters))
    }
  }
  true.thetas = matrix(ncol=nreps, nrow=np)
  
  for (i in 1:nreps){
    #sample data
    X = do.call(sample.x, c(list(niters=niters, np=np), x.control))
    theta = do.call(sample.theta, c(list(np=np), theta.control))
    Y = do.call(sample.y, c(list(mu=family$linkinv(X%*%theta)), y.control))
    true.thetas[, i] = theta
    for (m in method){
      for (l in learning.rate){
        result = implicit.fit(x=X, y=Y, family=family, intercept = F,
                              method=m, lr.type=l)
        estimates[[m]][[l]][i,,] = result$estimates
      }
    }
  }
  
 # get the trace of covariance matrix
 var = list()
 mean.estimates = list()
 for (m in method){
   for (l in learning.rate){
     var[[m]][[l]] = array(dim=c(niters))
     mean.estimates[[m]][[l]] = matrix(nrow=np, ncol=niters)
   }
 }
 for (m in method){
   for (l in learning.rate){
     for (j in 1:niters){
       iter.estimate = matrix(ncol=nreps, nrow=np)
       for (i in 1:nreps){
         iter.estimate[,i] = estimates[[m]][[l]][i, , j]
       }
       iter.cov = cov(iter.estimate)
       iter.cov = sum(diag(iter.cov))
       var[[m]][[l]][j] = iter.cov
       mean.estimates[[m]][[l]][, j] = rowSums(iter.estimate)/nreps
     }
   }
 }
 if (plot){
   par(mfrow=c(length(method), length(learning.rate)))
   for (m in method){
     for (l in learning.rate){
       plot(var[[m]][[l]], xlab='iter', ylab='variance')
       title(paste(m, l))
     }
   }
   par(mfrow=c(1,1))
 }
 list(variance = var, true.thetas = true.thetas, mean.estimates = mean.estimates, estimates = estimates)
}