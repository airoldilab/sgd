## Code for M-estimation.

rm(list=ls())
library(mvtnorm)
library(ggplot2)

l2 <- function(x) sqrt(sum(x**2))

generate.data <- function(N, p) {
  X = rmvnorm(N, mean=rep(0, p), sigma=diag(p)/N)
  thetastar = runif(p)
  thetastar = thetastar * 6 *sqrt(p) / l2(thetastar)
  
  ## noise
  ind = rbinom(N, size=1, prob=.95)
  epsilon = ind * rnorm(N) + (1-ind) * rep(10 ,N)
  
  
  Y = X %*% thetastar + epsilon
  return(list(y=Y, X=X, thetastar=thetastar))
}

algo.implicit <- function(thetan, xn, yn, lprime, gamma_n) {
  eta.n = sum(xn * thetan)
  xn.norm = l2(xn)**2
  ## Start algorithm.
  rn = gamma_n * lprime(eta.n, yn)
  Bn = c(0, rn)
  if(rn < 0)
    Bn <- c(rn, 0)
 
  if(Bn[1]==Bn[2]) {
    return(Bn[1])
  }
  
  implicit.eq <- function(x) {
    x - gamma_n * lprime(eta.n + x * xn.norm, yn)
  }
 
  ksi = uniroot(implicit.eq, lower=Bn[1], upper=Bn[2])$root
  lambda_n = rn / ksi
  #
  # lambda_n = 1 / (1+gamma_n * sum(xn**2))
  thetan + gamma_n * lambda_n * lprime(eta.n, yn) * xn
}

Mestimation.sgd <- function(data, niters=1e3, C=1, verbose=F) {
  X = data$X
  y = data$y
  thetastar = data$thetastar
  mse = c()
  p = ncol(X)
  N = nrow(X)
  theta.im = rep(0, p)
  # warning("running explicit SGD.")
  #
  lprime.huber = function(eta, y, lam=3) {
    z = y - eta
    if(abs(z) < lam) {
      return(z)
    } else {
      if(z > 0) return(lam)
      return(-lam)
    }
  }
  lprime.normal <- function(eta, y) {
    y-eta
  }
  
  plot.points = sample(1:niters, sqrt(niters)/2, replace=F)
  theta.bar = rep(0, p)
  #warning("Using sqrt(n) + averaging.")
  for(n in 1:niters) {
    j = sample(N, size=1)
    gamma_n = C/sqrt(n)
    xj = X[j, ]
    yj = y[j, ]
    # Do the implicit update.
    eta.j = sum(xj* theta.im)
    # theta.im = algo.implicit(theta.im, xj, yj, lprime.huber, gamma_n=gamma_n)
    theta.im = theta.im + gamma_n  * lprime.huber(eta.j, yj) * xj
    # theta.bar = (1/n) * ((n-1) * theta.bar + theta.im)
    mse = c(mse, l2(theta.im - thetastar)/sqrt(p))
    if(verbose && n %in% plot.points) {
      plot(mse, type="l", ylim=c(0, max(mse)), main=sprintf("mse=%.3f", tail(mse, 1)))
      abline(h=1.5, lty=3, col="green")
    }
  }
  # plot(mse, type="l", ylim=c(0, max(mse)))
  return(mse)
}

run.m_estimation.many <- function(nreps, nsamples=1e3, niters=1e4) {
  MSE = matrix(0, nrow=nreps, ncol=niters)
  pb = txtProgressBar(style=3)
  for(j in 1:nreps) {
    d = generate.data(N=nsamples, p=200)
    MSE[j,] <- Mestimation.sgd(d, niters=niters, C=15)
    setTxtProgressBar(pb, value=j/nreps)
  }
  ## Plotting.
  get.quant <- function(q) {
    apply(MSE, 2, function(x) quantile(x, q))
  }
  mse.med = get.quant(0.5)
  mse.1 = get.quant(0.05)
  mse.2 = get.quant(0.95)
  
  A = data.frame(mse=mse.med, mse.low=mse.1, mse.high=mse.2)
  xlabel <- "Iteration"
  ylabel <- "Mean squared error"
  
  p <- ggplot(data=A, aes(x=1:length(mse), y=mse, ymin=mse.low, ymax=mse.high)) + 
    geom_line(col="black") + 
    geom_ribbon(fill="cyan", alpha=0.5) + 
    geom_hline(yintercept=1.5, lty=3) +
#     scale_x_log10() + 
    #scale_y_log10() + 
    xlab(xlabel) + 
    ylab(ylabel)
  
  plot(p)
  # abline(h=1.5, lty=3, col="green")
  #ggsave(p, file="CC-vs_nCC_kT_prof.pdf", width=8, height=4.5)
}

main <- function() {
 
  d = generate.data(N=1e3, p=200)
  Mestimation.sgd(d, niters=1e4, C=15)
}
