## Model: Cox proportional hazards.
#
#  Example usage:
#     d  = generate.data(1e3, 10)
#     fit.cox(d)  # uses R package to find MLE.
#     cox.sgd(d, niters=1e4, implicit=T, average=T)
rm(list=ls())
library(survival)
library(glmnet)


generate.data <- function(n, p) {
  X = matrix(rbinom(n * p, size=2, prob = 0.1), nrow=n, ncol=p)
  # X[, 1] <- 1
  beta = exp(-(1/sqrt(p)) * seq(1, p))
  pred = apply(X, 1, function(r) exp(sum(r * beta)))
  Y = rexp(n, rate = pred)
  
  q3 = quantile(Y, prob=c(0.8))  # Q3 of Y
  epsilon = 0.001 # probability of censoring smallest Y
  k = log(1/epsilon - 1) / (q3 - min(Y))
  censor.prob = (1 + exp(-k * (Y-q3)))**(-1)
  # plot(censor.prob, main="Censoring probabilities", type="l")
  C = rbinom(n, size=1, prob= censor.prob)
  M = matrix(0, nrow=n, ncol=2)
  colnames(M) <- c("time", "status")
  M[, 1] <- Y
  M[, 2] <- C
  return(list(x=X, y=Y, censor=C, M=M, beta=beta))
}

dist <- function(x, y)  {
 sqrt(mean((x-y)**2))  
}

fit.cox <- function(data, verbose=T) {
  fit <- coxph(Surv(y, censor) ~ x, data)
  if(verbose) {
    print(names(fit))
    print(summary(fit))
    print("real parameters")
    print(data$beta)
    print("Distance")
    print(dist(fit$coeff, data$beta))
  }
  return(as.numeric(coefficients(fit)))
}

cox.sgd <- function(data, niters=1e3, C=1, implicit=F, averaging=F) {
  par(mfrow=c(1, 1))
  # Fit the parameter through other method, then use as "best".
  mse.best = dist(fit.cox(data), data$beta)
  if(implicit) {
    print("Running Implicit SGD for Cox PH model.")
  }
  # input
  n = length(data$y)
  p = ncol(data$x)
  beta = matrix(0, nrow=p, ncol=1)
  gammas = C / seq(1, niters)
  if(averaging) {
    gammas = C / seq(1, niters)**(1/3)
  }
  print(summary(gammas))
  betas = matrix(0, nrow=p, ncol=0)
  mse = c()
  
  # order units based on event times.
  ord = order(data$y)
  d = data$censor[ord]  # censor
  x = matrix(data$x[ord, ], ncol=p)  # ordered covariates.
  
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
  for(i in 1:niters) {
    gamma_i = gammas[i]
    ksi = exp(x %*% beta)
    j = units.sample[i] # sample unit
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
    if(dist(beta, rep(0, length(beta))) > 1e1) {
      stop("Possible divergence")
    }
    if(averaging) {
      beta.bar = (1/i) * ((i-1) * beta.bar + beta)
      mse <- c(mse, dist(beta.bar, data$beta))
    } else {
      mse <- c(mse, dist(beta, data$beta))
    }
    
    if(i %in% plot.points) {
     print(sprintf("Last MSE = %.3f", tail(mse, 1)))
      plot(mse, type="l", main=sprintf("dist=%.4f (implicit=%d)", tail(mse, 1), implicit), ylim=c(0, max(mse)))
     abline(h=mse.best, col="red", lty=3)
    }
    setTxtProgressBar(pb, value=i/niters)
  }
  print("SGD params")
  print(as.numeric(beta))
  print("Distance of last sgd iterate")
  print(dist(beta, data$beta))
  par(mfrow=c(1, 2))
  plot(mse, type="l", main=sprintf("dist=%.4f (implicit=%d)", tail(mse, 1), implicit), ylim=c(0, max(mse)))
  abline(h=mse.best, col="red", lty=3)
  plot( data$beta, as.numeric(beta), pch="x")
  lines(data$beta, data$beta, lty=3)
}

example.usage <- function() {
  print("Running example for Cox proportional hazards model.")
  # case 1: problematic.
#   d = generate.data(1e4, 50)
#   cox.sgd(d, niters = 1e4, C = 1, implicit = T, averaging=T)
  d = generate.data(1e4, 20)
   cox.sgd(d, niters = 5e4, C = 2, implicit = T, averaging=T)
}

glmnet.example <- function() {
  #Cox
  set.seed(10101)
  
  N=10000; p=100
  nzc=p
  x=matrix(rnorm(N*p),N,p)
  beta=rnorm(nzc)
  fx=x[,seq(nzc)]%*%beta
  hx=exp(fx)
  ty=rexp(N,hx)
  tcens=rbinom(n=N,prob=.3,size=1)# censoring indicator
  y=cbind(time=ty,status=1-tcens) # y=Surv(ty,1-tcens) with library(survival)

  fit=glmnet(x,y,family="cox", nlambda = 5)
  
  plot(fit)
  mse = as.numeric(apply(fit$beta, 2, function(col) dist(col, beta)))
  # print(length(mse))
  # print(names(fit))
  # print(dim(fit$beta))
  beta.glmnet = as.numeric(fit$beta[, which.min(mse)])
  # print("True beta")
  # print(beta)
  # print(beta.glmnet)
  print(sprintf("Min MSE for coxnet = %.3f", min(mse)))
  print(length(y[,1]))
  print(dim(x))

  d = list(x=x, M=y, y=as.numeric(y[,1]), 
           censor=as.numeric(y[,2]), 
           beta=beta)
  rm(y)
 # d = generate.data(N,p)
 # fit.cox(d)
  # coxph(Surv(y, censor) ~ x, data=d) 
 print("MSE of coxnet")
 print(summary(mse))
  cox.sgd(d, niters=1e5, C = .06, implicit = F, averaging=T)
   print("MSE of coxnet")
 print(summary(mse))
}