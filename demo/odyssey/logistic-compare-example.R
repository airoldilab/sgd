# Taken from:
# http://stackoverflow.com/questions/19532651/benchmarking-logistic-regression-using-glm-fit-bigglm-speedglm-glmnet-libli
testGLMResults_and_speed <- function(N, p, chunk=NULL, Ifsample=FALSE, size=NULL, reps=5){

  library(LiblineaR)
  library(speedglm)
  library(biglm)
  library(glmnet)

  # simulate dataset
  X = scale(matrix(rnorm(N*p), nrow=N, ncol=p))
  X1 = cbind(rep(1, N), X)
  q = as.integer(p/2)
  b = c(rnorm(q+1), rnorm(p-q)*10)
  eta = X1 %*% b

  # simulate Y
  simy <- function(x){
    p = 1/(1 + exp(-eta[x]))
    u = runif(1, 0, 1)
    return(ifelse(u<=p, 1, 0))
  }

  Y = sapply(1:N, simy)
  XYData = as.data.frame(cbind(y=Y, X))

  getSample <- function(X, Y=NULL, size){
    ix = sample(dim(X)[1], size, replace=FALSE)
    return(list(X=X[ix,], Y=Y[ix]))
  }

  #LiblineaR function
  fL <- function(X, Y, type, Ifsample=Ifsample, size=size){
    if(Ifsample){
      res = getSample(X, Y, size)
      X = res$X; Y = res$Y;
    }
    resL = LiblineaR(data=X, labels=Y, type=type)
    return(resL$W)
  }

  #glmnet
  fNet <- function(X, Y, Ifsample=Ifsample, size=size){
    if(Ifsample){
      res = getSample(X, Y, size)
      X = res$X; Y = res$Y;
    }
    resNGLM <- glmnet(x=X, y=Y, family="binomial", standardize=FALSE, type.logistic="modified.Newton")
    return(c(resNGLM$beta[, resNGLM$dim[2]], 0))
  }

  #glm.fit
  fglmfit <- function(X1, Y, Ifsample=Ifsample, size=size){
    if(Ifsample){
      res = getSample(X1, Y, size)
      X1 = res$X; Y=res$Y;
    }
    resGLM = glm.fit(x=X1, y=Y, family=binomial(link=logit))
    return(c(resGLM$coefficients[2:(p+1)], resGLM$coefficients[1]))
  }

  #speedglm
  fspeedglm <- function(X1, Y, Ifsample=Ifsample, size=size){
    if(Ifsample){
      res = getSample(X1, Y, size)
      X1 = res$X; Y=res$Y;
    }
    resSGLM = speedglm.wfit(y=Y, X=X1, family=binomial(link=logit), row.chunk=chunk)
    return(c(resSGLM$coefficients[2:(p+1)], resSGLM$coefficients[1]))
  }

  #bigglm
  fbigglm <- function(form, XYdf, Ifsample=Ifsample, size=size){
    if(Ifsample){
      res = getSample(XYdf, Y=NULL, size)
      XYdf = res$X;
    }
    resBGLM <- bigglm(formula=form, data=XYdf, family = binomial(link=logit), maxit=20)
    if(resBGLM$converged){
      resBGLM = summary(resBGLM)$mat[,1]
    } else {
      resBGLM = rep(-99, p+1)
    }
    return(c(resBGLM[2:(p+1)], resBGLM[1]))
  }

  ## benchmarking function
  ## calls reps times and averages parameter values and times over reps runs
  bench_mark <- function(func, args, reps){
    oneRun <- function(x, func, args){
      times = system.time(res <- do.call(func, args))[c("user.self", "sys.self", "elapsed")]
      return(list(times=times, res=res))
    }
    out = lapply(1:reps, oneRun, func, args)
    out.times = colMeans(t(sapply(1:reps, function(x) return(out[[x]]$times))))
    out.betas = colMeans(t(sapply(1:reps, function(x) return(out[[x]]$res))))
    return(list(times=out.times, betas=out.betas))
  }

  #benchmark LiblineaR
  func="fL"
  args = list(X=X, Y=Y, type=0, Ifsample=Ifsample, size=size)
  res_L0 = bench_mark(func, args, reps)
  args = list(X=X, Y=Y, type=6, Ifsample=Ifsample, size=size)
  res_L6 = bench_mark(func, args, reps)

  #benchmark glmnet
  func = "fNet"
  args = list(X=X, Y=Y, Ifsample=Ifsample, size=size)
  res_GLMNet = bench_mark(func, args, reps)

  func="fglmfit"
  args = list(X1=X1, Y=Y, Ifsample=Ifsample, size=size)
  res_GLM = bench_mark(func, args, reps)

  func="fspeedglm"
  args = list(X1=X1, Y=Y, Ifsample=Ifsample, size=size)
  res_SGLM = bench_mark(func, args, reps)

  func = "fbigglm"
  # create formula for bigglm
  xvarname = paste("V", 2:dim(XYData)[2], sep="")
  f  = as.formula(paste("y ~ ", paste(xvarname, collapse="+")))
  args = list(form=f, XYdf=XYData, Ifsample=Ifsample, size=size)
  res_BIGGLM = bench_mark(func, args, reps)

  summarize <- function(var){
    return(rbind(L0=res_L0[[var]], L6=res_L6[[var]],
                GLMNet=res_GLMNet[[var]], GLMfit=res_GLM[[var]],
                speedGLM=res_SGLM[[var]], bigGLM=res_BIGGLM[[var]]))
  }

  times = summarize("times")
  betas = rbind(summarize("betas"), betaTRUE = c(b[2:(p+1)], b[1]))
  colnames(betas)[dim(betas)[2]] = "Bias"
  # compare betas with true beta
  betacompare = t(sapply(1:dim(betas)[1], function(x) betas[x,]/betas[dim(betas)[1],]))


  print(paste("Run times averaged over", reps, "runs"))
  print(times)

  print(paste("Beta estimates averaged over", reps, "runs"))
  print(betas)

  print(paste("Ratio Beta estimates averaged over", reps, "runs for all methods (reference is true beta)"))
  print(betacompare)
}
testGLMResults_and_speed(10000, 5, 500, FALSE, 5000, 5)
