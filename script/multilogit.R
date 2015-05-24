library(sgd)
library(ggplot2)
source("load_mnist.R")
multilogit.fit <- function(X, y, ...){
  
  labels <- unique(y)
  nlabels <- length(unique(y))
  pivot.label <- labels[1]
  coefs <- array(0, dim=c(nlabels, ncol(X)+1, 100))
  pos <- array(0, dim=c(nlabels-1, 100))
  count <- 2
  for (label in labels[2:nlabels]){
    X_temp <- X[y==label | y==pivot.label, ]
    y_temp <- y[y==label | y==pivot.label]
    select <- y_temp==label
    y_temp[!select] <- 0
    y_temp[select] <- 1
    X_temp <- cbind(rep(1, nrow(X_temp)), X_temp)
    model <- sgd(X_temp, y_temp, "glm", model.control=list(family="binomial"), ...)
    coefs[count, , ] <- model$estimates
    pos[count-1, ] <- model$pos
    count <- count + 1
  }
  return(list(coefs=coefs, pos=pos, labels=labels))
}

multilogit.predict <- function(model, X){
  X <- cbind(rep(1, nrow(X)), X)
  etas <- array(0, dim=c(dim(model$coefs)[1], nrow(X), dim(model$coefs)[3]))
  # TODO: vectorize this
  for (i in 1:dim(model$coefs)[1]){
    etas[i, , ] <- X %*% model$coefs[i, , ]
  }
  # TODO: vectorize this
  pred <- apply(etas, c(2,3), function(x) model$labels[which.max(x)])
  return(list(pred=pred, pos=model$pos))
}

plot.error <- function(preds, ys, names){
  dat = data.frame()
  count <- 1
  for (pred in preds){
    error <- 1 - colSums(pred$pred == ys[[count]]) / nrow(pred$pred)
    pos <- colMeans(pred$pos)
    temp_dat <- data.frame(error=error, pos=pos)
    temp_dat[["label"]] <- as.factor(names[[count]])
    dat <- rbind(dat, temp_dat)
    count <- count + 1
  }
  p <- ggplot2::ggplot(dat, ggplot2::aes(x=pos, y=error, group=label)) +
    ggplot2::geom_line(ggplot2::aes(linetype=label)) +
    ggplot2::theme_bw() +
    ggplot2::theme(
      panel.border=ggplot2::element_blank(),
      panel.grid.major=ggplot2::element_blank(),
      panel.grid.minor=ggplot2::element_blank(),
      axis.line=ggplot2::element_line(color="black"),
      legend.position=c(1, 1),
      legend.justification = c(1, 1),
      legend.title=ggplot2::element_blank(), 
      legend.key=ggplot2::element_blank(),
      legend.background=ggplot2::element_rect(linetype="solid", color="black")
    ) +
    #ggplot2::scale_x_log10() +
    #ggplot2::scale_y_log10() +
    ggplot2::labs(
      title="Error",
      x="log-Iteration",
      y="log-Error"
    )
  return(p)
}

X = train$x[train$y==0 | train$y==1 | train$y==2,]
y = train$y[train$y==0 | train$y==1 | train$y==2]
tX = test$x[test$y==0 | test$y==1 | test$y==2,]
ty = test$y[test$y==0 | test$y==1 | test$y==2]
model <- multilogit.fit(X, y)
pred <- multilogit.predict(model, tX) 
plot.error(list(pred), list(ty), "1")


# 
# model <- multilogit.fit(train$x, train$y)
# pred <- multilogit.predict(model, test$x) 
# plot.error(list(pred), list(test$y), "1")

