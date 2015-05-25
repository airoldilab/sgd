source("load_mnist.R")
source("multilogit.R")
source("plot.R")


run_exp <- function(methods, lrs, np, X, y, X_test, y_test, plot=T){
  models = list()
  preds = list()
  y_tests = list()
  for (i in 1:length(methods)){
    model <- multilogit.fit(X, y, sgd.control=list(
      method=methods[[i]], lr=lrs[[i]], npasses=np[[i]]))
    pred <- multilogit.predict(model, X_test) 
    models[[i]] <- model
    preds[[i]] <- pred
    y_tests[[i]] <- y_test
    print(sprintf("%d experiment done!", i))
  }
  if (plot){
    return(plot.error(preds, y_tests, methods))
  } else{
    return(list(models=models, preds=preds))
  }
}

dat = load_mnist()
X = dat$train$x[1:10000, ]
y = dat$train$y[1:10000]
X_test = dat$test$x[1:2000, ]
y_test = dat$test$y[1:2000]


methods = list("sgd", "ai-sgd", "implicit")
lrs = list("adagrad", "one-dim", "one-dim")
np = list(3, 3, 3)

methods = list("ai-sgd")
lrs = list("one-dim")
np = list(10)
run_exp(methods, lrs, np, X, y, X_test, y_test)




X = train$x[train$y==0 | train$y==1 | train$y==2,]
y = train$y[train$y==0 | train$y==1 | train$y==2]
tX = test$x[test$y==0 | test$y==1 | test$y==2,]
ty = test$y[test$y==0 | test$y==1 | test$y==2]
model <- multilogit.fit(X, y, sgd.control=list(method="ai-sgd", npasses=100))
pred <- multilogit.predict(model, tX) 
plot.error(list(pred), list(ty), "1")


model <- multilogit.fit(train$x, train$y, sgd.control=list(method="implicit"))
pred <- multilogit.predict(model, test$x) 
plot.error(list(pred), list(test$y), "1")

coef=as.matrix(read.csv("coef.csv", sep = ",", header = F))
coefs = array(0, dim = c(3, 785, 100))
pos = array(0, dim=c(2, 100))
for (i in 1:100){
  coefs[,,i] = coef
  pos[,i] = i
}
labels = c(0,1,2)
model = list(coefs=coefs, pos=pos, labels=labels)


