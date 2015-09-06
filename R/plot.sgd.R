#' Plot objects of class \code{sgd}.
#'
#' @param x object of class \code{sgd}.
#' @param \dots
#' @param type character specifying the type of plot: \code{"mse"},
#'   \code{"runtime"}
#'
#' @export
plot.sgd <- function(x, ..., type="mse") {
  if (type == "mse") {
    plot <- plot_mse
  } else if (type %in% c("runtime", "mse-runtime")) {
    plot <- function(x, ..., xaxis="runtime") plot_mse(x, ..., xaxis=xaxis)
  } else {
    stop("'type' not recognized")
  }
  return(plot(x, ...))
}

################################################################################
# Auxiliary functions: plots
################################################################################

get_mse_glm <- function(x, x_test, y_test) {
  nests <- ncol(x$estimates)
  eta <- x_test %*% x$estimates # assuming intercepts in X
  mu <- x$model.out$family$linkinv(eta)
  mse <- rep(NA, nests)
  for (j in 1:nests) {
    mse[j] <- colMeans((mu[, j] - y_test)^2)
  }
  return(mse)
}

get_mse_m <- function(x, x_test, y_test) {
  nests <- ncol(x$estimates)
  eta <- x_test %*% x$estimates # assuming intercepts in X
  mu <- eta
  mse <- rep(NA, nests)
  for (j in 1:nests) {
    mse[j] <- colMeans((mu[, j] - y_test)^2)
  }
  return(mse)
}

# TODO in the same way as plot.R, allow for a list of sgd objects
plot_mse <- function(x, x_test, y_test, xaxis="iter") {
  if (x$model %in% c("lm", "glm")) {
    get_mse <- get_mse_glm
  } else if (x$model == "m") {
    get_mse <- get_mse_m
  # TODO
  } else {
    stop("'model' not recognized")
  }
  #sgds <- list(x, ...)
  dat <- data.frame()
  count <- 1
  #for (sgd in sgds) {
  sgd <- x
    mse <- get_mse(sgd, x_test, y_test)
    temp_dat <- data.frame(mse=mse,
                           time=sgd$time,
                           pos=sgd$pos,
                           label=count)
    temp_dat <- temp_dat[!duplicated(temp_dat$pos), ]
    dat <- rbind(dat, temp_dat)
    count <- count + 1
  #}
  dat$label <- as.factor(dat$label)

  # TODO there's got to be a more efficient way to do this...
  if (xaxis == "iter") {
    p <- ggplot2::ggplot(dat, ggplot2::aes(x=pos, y=mse, group=label))
  } else if (xaxis == "runtime") {
    p <- ggplot2::ggplot(dat, ggplot2::aes(x=time, y=mse, group=label))
  } else {
    stop("'xaxis' not recognized")
  }
  p <- p + ggplot2::geom_line(ggplot2::aes(linetype=label, color=label)) +
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
    ggplot2::scale_y_log10()
  if (xaxis == "iter") {
    p <- p +
      ggplot2::scale_x_log10(
        breaks=10^(1:log(sgd.theta$pos[length(sgd.theta$pos)], base=10))) +
      ggplot2::labs(
        title="Mean Squared Error",
        x="log-Iteration",
        y="log-MSE"
      )
  } else if (xaxis == "runtime") {
    p <- p +
      #ggplot2::scale_x_continuous(
      #  breaks=10^(1:log(sgd.theta$pos[length(sgd.theta$pos)], base=10))) +
      ggplot2::labs(
        title="Mean Squared Error",
        x="Runtime (s)",
        y="log-MSE"
      )
  } else {
    stop("'xaxis' not recognized")
  }

  return(p)
}
