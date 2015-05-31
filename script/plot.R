library(sgd)
library(ggplot2)

plot.sgd <- function(sgds, names, type="mse") {
  # Plot multiple sgd objects
  # Args:
  #  sgds: a list of sgd objects
  #  names: a list of names for sgds; the name will appear the legend

  if (type == "mse") {
    plot <- plot_mse
  } else {
    print(type)
    stop("'type' not recognized")
  }
  return(plot(sgds, names))
}

get_mse_glm <- function(x) {
  eta <- x$sample.x %*% x$estimates
  mu <- x$family$linkinv(eta)
  mse <- colMeans((mu - x$sample.y)^2)
  return(mse)
}

plot_mse <- function(sgds, names, np) {

  # Plot training MSE
  # Args:
  #   sgds: a list of sgd objects
  #   names: a list of names that will be included in the legend

  if (any(class(sgds[[1]]) %in% "glm")) {
    get_mse <- get_mse_glm
  } else {
    stop("Model not recognized!")
  }

  dat <- data.frame()
  count <- 1
  for (sgd in sgds) {
    mse <- get_mse(sgd)
    temp_dat <- data.frame(mse=mse, pos=sgd$pos[1, ])
    temp_dat <- temp_dat[!duplicated(temp_dat$pos), ]
    temp_dat[["npass"]] <- temp_dat$pos/max(temp_dat$pos) * np[[count]]
    temp_dat[["label"]] <- as.factor(names[[count]])
    dat <- rbind(dat, temp_dat)
    count <- count + 1
  }

  pos <- 0
  label <- 0
  p <- ggplot2::ggplot(dat, ggplot2::aes(x=npass, y=mse, group=label)) +
    ggplot2::geom_line(ggplot2::aes(linetype=label, color=label)) +
    ggplot2::theme(
      panel.background=ggplot2::element_blank(),
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
    ggplot2::scale_fill_hue(l=50) +
    ggplot2::scale_x_continuous(limits=c(0, max(unlist(np))), breaks=seq(0.5, max(unlist(np)), 0.5)) +
    ggplot2::scale_y_continuous(breaks=seq(0.05, 1, 0.05)) +
    ggplot2::labs(
      title="Mean Squared Error",
      x="Number of passes",
      y=""
    )
  return(p)
}

plot.error <- function(preds, ys, names, np, title) {

  # Plot test error for classification
  # Args:
  #   preds: a list of prediction objects.
  #     each prediction object is a list of pred and pos
  #       pred: nsamples * niter
  #       pos: ? * niter
  #   ys: a list of true labels
  #     each y: length(y) == nsamples
  #   names: a list of names that will be included in the legend
  #   title: title of plot

  dat <- data.frame()
  count <- 1
  for (pred in preds) {
    error <- 1 - colSums(pred$pred == ys[[count]]) / nrow(pred$pred)
    pos <- colMeans(pred$pos)
    temp_dat <- data.frame(error=error, pos=pos)
    temp_dat[["npass"]] <- temp_dat$pos/max(temp_dat$pos) * np[[count]]
    temp_dat[["label"]] <- as.factor(names[[count]])
    dat <- rbind(dat, temp_dat)
    count <- count + 1
  }
  ylimits <- mean(dat$error) + c(-2.5*sd(dat$error), 2*sd(dat$error))
  p <- ggplot2::ggplot(dat, ggplot2::aes(x=npass, y=error, group=label)) +
    ggplot2::geom_line(ggplot2::aes(linetype=label, color=label)) +
    ggplot2::theme(
      panel.background=ggplot2::element_blank(),
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
    ggplot2::scale_fill_hue(l=50) +
    ggplot2::scale_x_continuous(limits=c(0, max(unlist(np))), breaks=seq(0.5, max(unlist(np)), 0.5)) +
    ggplot2::scale_y_continuous(limits=ylimits, breaks=seq(0.05, 1, 0.05)) +
    ggplot2::labs(
      title=title,
      x="Number of passes",
      y=""
    )
  return(p)
}

plot.error.runtime <- function(preds, ys, names, times, title) {
  # Plot test error for classification by runtime
  #
  # Args:
  #   preds: a list of prediction objects.
  #     each prediction object is a list of pred and pos
  #       pred: nsamples * niter
  #       pos: ? * niter
  #   ys: a list of true labels
  #     each y: length(y) == nsamples
  #   names: a list of names that will be included in the legend
  #   times: a list of times
  dat <- data.frame()
  count <- 1
  for (pred in preds) {
    error <- 1 - colSums(pred$pred == ys[[count]]) / nrow(pred$pred)
    pos <- colMeans(pred$pos)
    temp_dat <- data.frame(error=error, pos=pos, time=times[[count]])
    temp_dat[["label"]] <- as.factor(names[[count]])
    dat <- rbind(dat, temp_dat)
    count <- count + 1
  }
  if (max(dat$time) < 5) {
    time_breaks <- seq(0, max(dat$time), 1)
  } else if (max(dat$time) < 10) {
    time_breaks <- seq(0, max(dat$time), 2)
  } else if (max(dat$times) < 50) {
    time_breaks <- seq(0, max(dat$time), 5)
  } else {
    time_breaks <- seq(0, max(dat$time), 10)
  }
  ylimits <- mean(dat$error) + c(-2.5*sd(dat$error), 2*sd(dat$error))
  p <- ggplot2::ggplot(dat, ggplot2::aes(x=time, y=error, group=label)) +
    ggplot2::geom_line(ggplot2::aes(linetype=label, color=label)) +
    ggplot2::theme(
      panel.background=ggplot2::element_blank(),
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
    ggplot2::scale_fill_hue(l=50) +
    #ggplot2::scale_x_continuous(limits=c(0, max(dat$time)), breaks=seq(0, max(dat$time), 1)) +
    ggplot2::scale_x_continuous(limits=c(min(dat$time), max(dat$time)),
    breaks=time_breaks) +
    ggplot2::scale_y_continuous(limits=ylimits, breaks=seq(0.05, 1, 0.05)) +
    ggplot2::labs(
      title=title,
      x="Training time (sec.)",
      y=""
    )
  return(p)
}

plot.cost <- function(preds, ys, names, np, title) {
  dat <- data.frame()
  count <- 1
  for (pred in preds) {
    predclass <- match(ys[[count]], pred$labels)
    logprob <- matrix(NA, nrow=dim(pred$prob)[2], ncol=dim(pred$prob)[3])
    for (i in 1:dim(pred$prob)[2]) {
      logprob[i, ] <- log(pred$prob[predclass[i], i, ])
    }
    logloss <- -colSums(logprob) / nrow(logprob)
    pos <- colMeans(pred$pos)
    temp_dat <- data.frame(logloss=logloss, pos=pos)
    temp_dat[["npass"]] <- temp_dat$pos/max(temp_dat$pos) * np[[count]]
    temp_dat[["label"]] <- as.factor(names[[count]])
    dat <- rbind(dat, temp_dat)
    count <- count + 1
  }
  ylimits <- mean(dat$logloss) + c(-2.5*sd(dat$logloss), 2*sd(dat$logloss))
  p <- ggplot2::ggplot(dat, ggplot2::aes(x=npass, y=logloss, group=label)) +
    ggplot2::geom_line(ggplot2::aes(linetype=label, color=label)) +
    ggplot2::theme(
      panel.background=ggplot2::element_blank(),
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
    ggplot2::scale_fill_hue(l=50) +
    ggplot2::scale_x_continuous(limits=c(0, max(unlist(np))), breaks=seq(0.5, max(unlist(np)), 0.5)) +
    ggplot2::scale_y_continuous(limits=ylimits) +
    ggplot2::labs(
      title=title,
      x="Number of passes",
      y=""
    )
  return(p)
}

