## Model: Cox proportional hazards.
#
# TODO(ptoulis): Integrate to main SGD code.
#
#
#   Data =    Y (obs times) |  X (covariates) |  Censor
#                2.25             (..)             0
#            
rm(list=ls())
library(survival)
library(glmnet)


genx = function(n,p,rho){
  #    generate x's multivariate normal with equal corr rho
  # Xi = b Z + Wi, and Z, Wi are independent normal.
  # Then Var(Xi) = b^2 + 1
  #  Cov(Xi, Xj) = b^2  and so cor(Xi, Xj) = b^2 / (1+b^2) = rho
  z=rnorm(n)
  if(abs(rho)<1){
    beta=sqrt(rho/(1-rho))
    x0=matrix(rnorm(n*p),ncol=p)
    A = matrix(z, nrow=n, ncol=p, byrow=F)
    x= beta * A + x0
  }
  if(abs(rho)==1){ x=matrix(z,nrow=n,ncol=p,byrow=F)}
  
  return(x)
}
dist <- function(x, y)  {
  sqrt(mean((x-y)**2))  
}

generate.data <- function(n, p, rho=0.2) {
  ## Generate data
  #   #
  #   # Returns:
  #   #   LIST(X, Y, censor, true.beta)
  #   #     X = Nxp matrix of covariates.
  #   #     Y = Nx1 vector of observed times.
  #   #     censor = Nx1 vector {0, 1} of censor indicators.
  #   #     true.beta = p-vector of true model parameters.
  #   #     M = (Y, censor) as matrix 
  X = genx(n, p, rho=0.)
  # X[, 1] <- 1
  beta = ((-1)^(1:p))*exp(-2*((1:p)-1)/20)
  pred = apply(X, 1, function(r) exp(sum(r * beta)))
  Y = rexp(n, rate = pred)
  
  q3 = quantile(Y, prob=c(0.8))  # Q3 of Y
  epsilon = 0.001 # probability of censoring smallest Y
  k = log(1/epsilon - 1) / (q3 - min(Y))
  censor.prob = (1 + exp(-k * (Y-q3)))**(-1)

  C = rbinom(n, size=1, prob= censor.prob)
  
  ## Order the data
  order.i = order(Y)
  X = X[order.i, ]
  Y = Y[order.i]
  C = C[order.i]
  
  M = matrix(0, nrow=n, ncol=2)
  colnames(M) <- c("time", "status")
  M[, 1] <- Y
  M[, 2] <- 1-C
  return(list(X=X, Y=Y, censor=C, M=M, true.beta=beta))
}


coxbatch <- function(data, verbose=T) {
  # Uses coxph function from survival packae
  # Not of interest because it does not scale.
  fit <- coxph(Surv(Y, censor) ~ X, data)
  if(verbose) {
    print(names(fit))
    print(summary(fit))
    par(mfrow=c(1, 1))
    plot(data$true.beta, as.numeric(fit$coeff), pch="x")
    lines(data$true, data$true.beta, lty=3, col="red")
    print("Distance")
    print(dist(fit$coeff, data$true.beta))
  }
  return(as.numeric(coefficients(fit)))
}

cox.sgd <- function(data, niters=1e3, C=1, implicit=F, averaging=F) {
  # Cox proportional hazards through SGD.
  # Args:
  #   data = list(X, Yt, censor, true.beta)
  #
  # TODO(ptoulis): Change code to do cross-validation.
  mse.best = dist(data$true.beta, coxbatch(data, verbose=F))
  if(implicit) {
    print("Running Implicit SGD for Cox PH model.")
  }
  #   input
  n = length(data$Y)
  p = ncol(data$X)
  beta = matrix(0, nrow=p, ncol=1)
  gammas = C / seq(1, niters)
  if(averaging) {
    gammas = C / seq(1, niters)**(1/3)
  }
 
  betas = matrix(0, nrow=p, ncol=0)
  mse = c()
  
  d = 1-data$censor  # failure observed.
  x = data$X  # ordered covariates.
  
  # plotting params.
  plot.points = as.integer(seq(1, niters, length.out=20))
  pb = txtProgressBar(style=3)
  
  # params for the implicit method.
  fj <- NA
  fim <- NA
  lam <- 1
  # parameter for averaging
  beta.bar <- matrix(0, nrow=p, ncol=1)
  
  units.sample = sample(1:n, size=niters, replace=T)
  for(iter in 1:niters) {
    
    gamma_i = gammas[iter]
    ksi = exp(x %*% beta)
   
    j = units.sample[iter] # sample unit
   # print(sprintf("Changing unit %d", j))
    Xj = matrix(x[j, ], ncol=1) # get covariates
    
    # baseline hazards for units in risk set Rj.
    Hj = sum(head(d, j) / head(rev(cumsum(rev(ksi))), j))  
  
    # Defined for the implicit
    # TODO(ptoulis): Numerical problem still exists. 
    #   beta params can still get very large, whereas Hj goes down. 
    #   Should we normalize?
    # 
    if(implicit) {
      Xj.norm = sum(Xj**2)
      fj = exp(sum(beta * Xj))
      Aj = Hj * fj
      dj = d[j] # censor data.
      Bj = gamma_i * Xj.norm * (dj - Aj)
      if(Aj==0 || Bj==0 || dj==Aj) {
        lam <- 1
      } else if(dj==0) {
        lam = uniroot(f = function(el) Bj * el - log(el), lower=0, upper=1)$root
      } else if(dj  > Aj) {
        rj = dj / Aj
        lam = uniroot(f = function(el) Bj * el - log(max(0, rj - (rj-1) * el)), 
                      lower=1e-10, upper=rj / (rj-1))$root
      } else if (dj < Aj) {
        rj = dj / Aj
        lam = uniroot(f = function(el) Bj * el - log(rj + (1-rj) * el), 
                      lower=1e-10, upper=1)$root
      }
    }
     
    # Update. (lam=1 for explicit -- updated for implicit)
    beta = beta + gamma_i * lam * (d[j] - Hj * ksi[j]) * Xj
    # beta = data$true.beta
    if(dist(beta, rep(0, length(beta))) > 1e1) {
      stop("Possible divergence")
    }
    
    if(averaging) {
      beta.bar = (1/iter) * ((iter-1) * beta.bar + beta)
      mse <- c(mse, dist(beta.bar, data$true.beta))
    } else {
      mse <- c(mse, dist(beta, data$true.beta))
    }
    
    # Plotting.
    if(iter %in% plot.points) {
     print(sprintf("Last MSE = %.3f", tail(mse, 1)))
      plot(mse, type="l", main=sprintf("dist=%.4f (implicit=%d)", tail(mse, 1), implicit), 
           ylim=c(mse.best/2, max(mse)))
     abline(h=mse.best, col="red", lty=3)
    }
    setTxtProgressBar(pb, value=iter/niters)
  }
  # Final printing/plotting
  # TODO(ptoulis): Remove once the function is finalized.
  print("SGD params")
  print(as.numeric(beta))
  print("Distance of last sgd iterate")
  print(dist(beta, data$true.beta))
  par(mfrow=c(1, 2))
  plot(mse, type="l", main=sprintf("dist=%.4f (implicit=%d)", tail(mse, 1), implicit), ylim=c(0, max(mse)))
  abline(h=mse.best, col="red", lty=3)
  plot(data$true.beta, data$true.beta, col="red", lty=3, type="l")
  points(data$true.beta, as.numeric(beta), pch="x")
}

coxnet <- function(data) {
  # Runs the elastic net on the Cox proportional hazards dataset
  #
  #
  X = data$X
  y_glmnet = cbind(time=data$Y, status=1-data$censor)
  fit = glmnet(X, y_glmnet, family="cox")
  # fit = (beta, lambda, ...)
  mse = apply(fit$beta, 2, function(b) dist(b, data$true.beta))
  
  ## Plotting
  par(mfrow=c(1, 1))
  mse.best = dist(data$true.beta, coxbatch(data, verbose = F))
  print(sprintf("MSE from coxph=%.3f", mse.best))
  plot(fit$lambda, mse, type="l", main="MSE of coxnet")
  abline(h=mse.best, col="red", lty=3)
  print(sprintf("min MSE from coxnet = %.3f", min(mse)))
}

sgd.vs.glmnet <- function(use.real.data=F) {
  # Runs glmnet vs. SGD for fitting simulated or real-world dataset.
  # 
  # TODO(ptoulis): In real-data, evaluate both based on cross-validation.
  # TODO(ptoulis): Compute the CV plots for both methods and datasets.
  #
  data = gen.data(N=1e3, p=20, rho = 0.2)

  # unload
  X = data$X
  Yt = data$Yt
  censor = data$censor
  beta.star = data$true.beta
  
  if(use.real.data) {
    attach("LymphomaData.rda")
    X = t(patient.data$x)
    Yt = patient.data$time
    status = patient.data$status
    censor = 1-status
  }
  
  y.glmnet = cbind(time=Yt, status=1-censor)
  print(head(y.glmnet))
  # 1. glmnet
  fit = glmnet(X, y.glmnet, family="cox")
  plot(fit)
  mse = as.numeric(apply(fit$beta, 2, function(b) sqrt(mean((b - beta.star)**2))))
  mse.best = min(mse)
  plot(fit$lambda, mse, type="l", lty=3)
  
  # data for SGD
  cox.sgd(data, niters=1e4, C=0.1, implicit = T, averaging=F)
}


