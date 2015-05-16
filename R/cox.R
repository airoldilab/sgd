## Model: Cox proportional hazards.
#
# TODO(ptoulis): Integrate to main SGD code.
rm(list=ls())
library(survival)
library(glmnet)


## Taken from Jerry Friedman.
gen.times = function(x, snr=10){
  # generate data according to Friedman's setup
  n=nrow(x)
  p=ncol(x)
  b=((-1)^(1:p))*exp(-2*((1:p)-1)/20)
  # b=sample(c(-0.8, -0.45, 0.45, 0.9, 1.4), size=p, replace=T)
  # ((-1)^(1:p))*(1:p)^{-0.65}#exp(-2*((1:p)-1)/20)
  f = x%*%b
  z = rnorm(n)
  k = sqrt(var(f)/(snr*var(z)))
  
  true.Y = exp(f + k*z)
  censor.times = exp(k * z)
  
  censor.ind = as.numeric(censor.times < true.Y)
  ytime = sapply(1:length(true.Y), function(i) min(true.Y[i], censor.times[i]))
  
  return(list(true.beta=b, y=cbind(time=ytime, 
                                   censor=censor.ind)))
}
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

gen.data <- function(N, p, rho=0.2) {
  ## Generate data (X=covariates, Yt=obs. times, censor ={0,1} indicators, beta.star=true.pars)
  X = genx(N, p, rho)
  times = gen.times(X)
  return(list(X=X, 
              Yt=times$y[,1], 
              censor=times$y[,2],
              true.beta=times$true.beta))
}


fit.cox <- function(data, verbose=T) {
  # Uses coxph function from survival packae
  # Not of interest because it does not scale.
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
  # Cox proportional hazards through SGD.
  # Args:
  #   data = list(X, Yt, censor, true.beta)
  #
  # TODO(ptoulis): Change code to do cross-validation.
  mse.best = 0.5
  par(mfrow=c(1, 1))
  if(implicit) {
    print("Running Implicit SGD for Cox PH model.")
  }
  # input
  n = length(data$Yt)
  p = ncol(data$X)
  beta = matrix(0, nrow=p, ncol=1)
  gammas = C / seq(1, niters)
  if(averaging) {
    gammas = C / seq(1, niters)**(1/3)
  }
  print(summary(gammas))
  betas = matrix(0, nrow=p, ncol=0)
  mse = c()
  
  # order units based on event times.
  ord = order(data$Yt)
  d = data$censor[ord]  # censor
  x = matrix(data$X[ord, ], ncol=p)  # ordered covariates.
  
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
      mse <- c(mse, dist(beta, data$true.beta))
    }
    
    if(i %in% plot.points) {
     print(sprintf("Last MSE = %.3f", tail(mse, 1)))
      plot(mse, type="l", main=sprintf("dist=%.4f (implicit=%d)", tail(mse, 1), implicit), 
           ylim=c(mse.best/2, max(mse)))
     abline(h=mse.best, col="red", lty=3)
    }
    setTxtProgressBar(pb, value=i/niters)
  }
  print("SGD params")
  print(as.numeric(beta))
  print("Distance of last sgd iterate")
  print(dist(beta, data$true.beta))
  par(mfrow=c(1, 2))
  plot(mse, type="l", main=sprintf("dist=%.4f (implicit=%d)", tail(mse, 1), implicit), ylim=c(0, max(mse)))
  abline(h=mse.best, col="red", lty=3)
  plot( data$true.beta, as.numeric(beta), pch="x")
  lines(data$true.beta, data$true.beta, lty=3)
}

sgd.vs.glmnet <- function(use.real.data=F) {
  # Runs glmnet vs. SGD for fitting simulated or real-world dataset.
  # 
  # TODO(ptoulis): Simulated does not give reasonable fit (all params=0)
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


