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
library(logging)
unlink("cox.log")
logReset()
# addHandler(writeToFile, file="cox.log")

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

log.lik <- function(data, beta) {
  X = data$X
  ksi = as.numeric(exp(X %*% beta))
  H = rev(cumsum(rev(ksi))) 
  d = 1 - data$censor
  sum(d * (log(ksi) - log(H)))
}

verify.mle.conditions <- function() {
  p = 10
  n = 1e3
  data = generate.data(n, p, rho = 0.5)
  beta.hat = coxbatch(data, verbose = F)
  print("Calculating MLE.") 
  beta.new = optim(par=rep(0, p), 
                   fn=function(b) -log.lik(data, b), 
                   method="L-BFGS")$par
  print("MLE params.")
  print(as.numeric(beta.new))
  print("True params")
  print(data$true.beta)
  print("Log-likelihood = ")
  print(log.lik(data, beta.new))
  
  ## 
  print("Verifying MLE conditions.")
  X = data$X
  ksi = as.numeric(exp(X %*% beta.new))
  Dh = diag(ksi)
  H = rev(cumsum(rev(ksi))) # Hi = xi_i + xi_i+1 + ...xi_n
  DH.inv = diag(1/H)
  L = lower.tri(DH.inv, diag = T) + 0

  I = diag(nrow(X))
  
  d = 1-data$censor
  A = t(X) %*% (I - Dh %*% L %*% DH.inv) %*% d
  if(sqrt(mean(A**2)) < 1e-1) {
    print(sprintf("Success! MLE conditions seem to hold. Max |A| =%.3f/ Min |A|=%.3f (all should be zero)", 
          max(abs(A)), min(abs(A))))
  } else {
    print("Failure. MLE conditions seem not to hold.")
  }
  # print(DH)
  
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
  
  X = genx(n, p, rho=rho)
  # rates.
  # rates = runif(n, min=1e-2, max=10)
  # Y = rexp(n, 
  # beta = solve(t(X) %*% X) %*% t(X) %*% log(rates)
  beta = 10  *((-1)^(1:p))*exp(-2*((1:p)-1)/20)
  # beta = 10 * seq(1, p)**(-0.5)
  # warning("Large coefficients")
  pred = exp(X %*% beta)
  Y = rexp(n, rate =pred)
  
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
  if(file.exists("cox.log")) unlink("cox.log")
  mse.best = dist(data$true.beta, coxbatch(data, verbose=F))
  if(implicit) {
    print("Running Implicit SGD for Cox PH model.")
  }
  #   input
  n = length(data$Y)
  p = ncol(data$X)
  loginfo(sprintf("n=%d instances and p=%d variables", n, p))
  beta = matrix(0, nrow=p, ncol=1)
  beta.new = beta
  
  gammas = C / seq(1, niters)
  if(averaging) {
    gammas = C / seq(1, niters)**(1/2)
  }
  loginfo(sprintf("Learning rates = %s", paste(head(gammas), collapse=", ")))
  
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
  
  units.sample = sample(which(d==1), size=niters, replace=T)
  for(iter in 1:niters) {
    
    # New iteration
    gamma_i = gammas[iter]
    ksi = exp(x %*% beta)
    ksi[is.infinite(ksi)] <- exp(100)
    
    j = units.sample[iter] # sample unit
    loginfo("=================================================================")
    loginfo(sprintf("Iteration = %d  gamma=%.3f --  j=%d, dj=%d  Yj=%.2f ", 
                    iter, gamma_i, j, d[j], data$Y[j]))
    loginfo(paste(round(beta, 2), collapse=", "))
    loginfo("ksi=")
    loginfo(paste(round(ksi, 2), collapse=","))
    # print(sprintf("Changing unit %d", j))
    Xj = matrix(x[j, ], ncol=1) # get covariates
    # using Adagrad rates.
  
    
    # baseline hazards for units in risk set Rj.
    Hj = sum(head(d, j) / head(rev(cumsum(rev(ksi))), j))  
    loginfo(sprintf("Hj = %.3f", Hj))
    # Defined for the implicit
    # TODO(ptoulis): Numerical problem still exists. 
    #   beta params can still get very large, whereas Hj goes down. 
    #   Should we normalize?
    # 
    if(implicit) {
      Xj.norm = sum(Xj**2)
      fj = exp(sum(beta * Xj))
      Aj = NA
      if(Hj==0) {
        Aj = 0
      } else {
        Aj = Hj * fj
      }
      # if(Aj < 1e-100) Aj <- 1e-100
      # if(Aj > 1e100) Aj <- 1e100
      
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
    ST_j = (d[j] - Hj * ksi[j])
    loginfo(sprintf("ST_j = %.3f", ST_j))
    loginfo(sprintf("Xj= %s", paste(round(Xj, 2), collapse=", ")))
    loginfo(sprintf("Update = %s", paste(round(gamma_i * lam * ST_j * Xj, 3), collapse=", ")))
    # Update
   
    beta.new = beta + gamma_i * lam * ST_j * Xj
    loginfo(sprintf("NEW beta = %s", paste(round(beta.new, 2), collapse=", ")))
    loginfo(sprintf("TRUE beta = %s", paste(round(data$true.beta, 2), collapse=", ")))
    loginfo(sprintf("NEW MSE = %.3f", dist(data$true.beta, beta.new)))
    # beta = data$true.beta
    if(dist(beta, rep(0, length(beta))) > 1e5) {
      print(as.numeric(beta))
      print(ST_j)
      print(as.numeric(beta.new))
      stop("Possible divergence")
    }
    beta <- beta.new
    
    if(averaging) {
      beta.bar = (1/iter) * ((iter-1) * beta.bar + beta)
      mse <- c(mse, dist(beta.bar, data$true.beta))
      if(tail(mse, 1) < mse.best) {
        print("Best beta. Stop?")
        print(as.numeric(beta.bar))
      }
    } else {
      mse <- c(mse, dist(beta, data$true.beta))
    }
    
    # Plotting.
    if(iter %in% plot.points) {
      print(sprintf("Last MSE = %.3f  (best=%.3f, gamma=%.2f, C*=%.3f)", 
                    tail(mse, 1), mse.best, gamma_i, 1))
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
  print("TRUE params")
  print(data$true.beta)
  print("Distance of last sgd iterate")
  print(dist(beta, data$true.beta))
  par(mfrow=c(1, 2))
  plot(mse, type="l", main=sprintf("dist=%.4f (implicit=%d)", tail(mse, 1), implicit), ylim=c(0, max(mse)))
  abline(h=mse.best, col="red", lty=3)
  plot(data$true.beta, data$true.beta, col="red", lty=3, type="l")
  points(data$true.beta, as.numeric(beta), pch="x")
}

cox.sgd2 <- function(data, niters=1e3, C=1, implicit=F, averaging=F) {
  # Cox proportional hazards through SGD.
  # Args:
  #   data = list(X, Yt, censor, true.beta)
  #
  # TODO(ptoulis): Change code to do cross-validation.
  
  mse.best = dist(data$true.beta, coxbatch(data, verbose=F))
  if(implicit) {
    print("Running Implicit SGD for Cox PH model.")
    print("")
  }
  #   input
  n = length(data$Y)
  p = ncol(data$X)
  loginfo(sprintf("n=%d instances and p=%d variables", n, p))
  beta = matrix(0, nrow=p, ncol=1)
  beta.new = beta
  
  gammas = C  / seq(1, niters)
  if(averaging) {
    gammas = C / seq(1, niters)**(1/2)
  }
  loginfo(sprintf("Learning rates = %s", paste(head(gammas), collapse=", ")))
  
  betas = matrix(0, nrow=p, ncol=0)
  mse = c()
  # plotting params.
  plot.points = as.integer(seq(1, niters, length.out=20))
  pb = txtProgressBar(style=3)
  # parameter for averaging
  beta.bar <- matrix(0, nrow=p, ncol=1)
  
  ## Data/definitions.
  d = 1-data$censor  # failure observed.
  X = data$X  # ordered covariates.
  L = lower.tri(diag(n), diag = T) + 0
  U = upper.tri(diag(n), diag=T) + 0
  I = diag(n)
  
  for(iter in 1:niters) {
    
    # New iteration
    gamma_i = gammas[iter]
    eta = X %*% beta
    ksi = as.numeric(exp(eta))
    D.ksi = diag(ksi)
    H.ksi = U %*% matrix(ksi, ncol=1) # Hi = xi_i + xi_i+1 + ...xi_n
    DH.inv = diag(as.numeric(1/H.ksi))
    # residual
    # r = (I - D.ksi %*% L %*% DH.inv) %*% d
    r = d - ksi * (L %*% (d/H.ksi))
    z = eta + r
    
    # quick SGD step to solve LMS -- avoid matrix inversions.
    j = sample(1:n, size=1)
    xj = matrix(X[j,], ncol=1)
    xj.norm = sum(xj**2)
    fctr = gamma_i / (1 + gamma_i * xj.norm)
    pred.j = sum(xj * beta)
    beta.new = NA
    if(implicit) {
      beta.new = beta - fctr * pred.j * xj + gamma_i * z[j] * (1 - fctr * xj.norm) * xj 
    } else {
      beta.new= beta + gamma_i * (z[j] - pred.j) * xj
    }
    
    beta <- beta.new
    if(dist(beta, rep(0, length(beta))) > 1e5) {
      print(as.numeric(beta))
        print(as.numeric(beta.new))
      stop("Possible divergence")
    }
    ## Update the old vector.
  
    
    if(averaging) {
      beta.bar = (1/iter) * ((iter-1) * beta.bar + beta)
      mse <- c(mse, dist(beta.bar, data$true.beta))
    } else {
      mse <- c(mse, dist(beta, data$true.beta))
    }
    
    # Plotting.
    if(iter %in% plot.points) {
      print(sprintf("Last MSE = %.3f  (best=%.3f, gamma=%.2f, C*=%.3f)", 
                    tail(mse, 1), mse.best, gamma_i, 1))
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
  print("TRUE params")
  print(data$true.beta)
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
  return(fit)
}

