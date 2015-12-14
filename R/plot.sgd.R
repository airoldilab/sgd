#' Plot objects of class \code{sgd}.
#'
#' @param x object of class \code{sgd}.
#' @param \dots additional arguments used for each type of plot. See
#'   \sQuote{Details}.
#' @param type character specifying the type of plot: \code{"mse"},
#'   \code{"clf"}, \code{"mse-param"}. See \sQuote{Details}. Default is
#'   \code{"mse"}.
#' @param xaxis character specifying the x-axis of plot: \code{"iteration"}
#'   plots the y values over the log-iteration of the algorithm;
#'   \code{"runtime"} plots the y values over the time in seconds to reach them.
#'   Default is \code{"iteration"}.
#'
#' @details
#' Types of plots available:
#' \describe{
#'   \item{\code{mse}}{Mean squared error in predictions, which takes the
#'     following arguments:
#'     \describe{
#'       \item{\code{x_test}}{test set}
#'       \item{\code{y_test}}{test responses to compare predictions to}
#'     }}
#'   \item{\code{clf}}{Classification error in predictions, which takes the
#'     following arguments:
#'     \describe{
#'       \item{\code{x_test}}{test set}
#'       \item{\code{y_test}}{test responses to compare predictions to}
#'     }}
#'   \item{\code{mse-param}}{Mean squared error in parameters, which takes the
#'     following arguments:
#'     \describe{
#'       \item{\code{true_param}}{true vector of parameters to compare to}
#'     }}
#' }
#'
#' @export
plot.sgd <- function(x, ..., type="mse", xaxis="iteration") {
  plot <- choose_plot(type, xaxis)
  return(plot(x, ...))
}

#' @export
#' @rdname plot.sgd
plot.list <- function(x, ..., type="mse", xaxis="iteration") {
  plot <- choose_plot(type, xaxis)
  return(plot(x, ...))
}

################################################################################
# Helper functions
################################################################################

choose_plot <- function(type, xaxis) {
  if (type == "mse") {
    if (xaxis == "iteration") {
      return(plot_mse)
    } else if (xaxis == "runtime") {
      return(function(x, ...) plot_mse(x, ..., xaxis="Runtime (s)"))
    }
  } else if (type == "mse-param") {
    if (xaxis == "iteration") {
      return(plot_mse_param)
    } else if (xaxis == "runtime") {
      return(function(x, ...) plot_mse_param(x, ..., xaxis="Runtime (s)"))
    }
  } else if (type == "clf") {
    if (xaxis == "iteration") {
      return(plot_clf)
    } else if (xaxis == "runtime") {
      return(function(x, ...) plot_clf(x, ..., xaxis="Runtime (s)"))
    }
  }
  stop("'type' not recognized")
}

get_mse <- function(x, x_test, y_test) {
  mu <- predict_all(x, x_test)
  nests <- ncol(mu)
  mse <- rep(NA, nests)
  for (j in 1:nests) {
    mse[j] <- mean((mu[, j] - y_test)^2)
  }
  return(mse)
}

get_mse_param <- function(x, true_param) {
  nests <- ncol(x$estimates)
  mse <- rep(NA, nests)
  for (j in 1:nests) {
    mse[j] <- mean((x$estimates[, j] - true_param)^2)
  }
  return(mse)
}

plot_mse <- function(x, x_test, y_test, label=1, xaxis="log-Iteration") {
  if (class(x) != "list") {
    x <- list(label=x)
  }
  dat <- data.frame()
  for (i in 1:length(x)) {
    mse <- get_mse(x[[i]], x_test, y_test)
    temp_dat <- data.frame(y=mse,
                           label=names(x)[i])
    if (xaxis == "log-Iteration") {
      temp_dat$x <- x[[i]]$pos
    } else if (xaxis == "Runtime (s)") {
      temp_dat$x <- x[[i]]$time
    }
    temp_dat <- temp_dat[!duplicated(temp_dat$x), ]
    dat <- rbind(dat, temp_dat)
  }
  dat$label <- as.factor(dat$label)

  p <- generic_plot(dat, xaxis) +
    ggplot2::scale_y_log10() +
    ggplot2::labs(
      title="Mean Squared Error",
      x=xaxis,
      y="")
  return(p)
}

plot_mse_param <- function(x, true_param, label=1, xaxis="log-Iteration") {
  if (class(x) != "list") {
    x <- list(x)
    names(x) <- label
  }
  dat <- data.frame()
  for (i in 1:length(x)) {
    mse <- get_mse_param(x[[i]], true_param)
    temp_dat <- data.frame(y=mse,
                           label=names(x)[i])
    if (xaxis == "log-Iteration") {
      temp_dat$x <- x[[i]]$pos
    } else if (xaxis == "Runtime (s)") {
      temp_dat$x <- x[[i]]$time
    }
    temp_dat <- temp_dat[!duplicated(temp_dat$x), ]
    dat <- rbind(dat, temp_dat)
  }
  dat$label <- as.factor(dat$label)

  p <- generic_plot(dat, xaxis) +
    ggplot2::scale_y_continuous(
      breaks=5 * 1:min((max(dat$y)/5), 100)) +
    ggplot2::labs(
      title="Mean Squared Error",
      x=xaxis,
      y="")
  return(p)
}

plot_clf <- function(x, x_test, y_test, label=1, xaxis="log-Iteration") {
  if (class(x) != "list") {
    x <- list(x)
    names(x) <- label
  }
  dat <- data.frame()
  for (i in 1:length(x)) {
    pred <- predict_all(x[[i]], x_test)
    pred <- (pred > 0.5) * 1
    error <- colSums(pred != y_test) / nrow(pred) # is this correct?
    temp_dat <- data.frame(y=error,
                           label=names(x)[i])
    if (xaxis == "log-Iteration") {
      temp_dat$x <- x[[i]]$pos
    } else if (xaxis == "Runtime (s)") {
      temp_dat$x <- x[[i]]$time
    }
    temp_dat <- temp_dat[!duplicated(temp_dat$x), ]
    dat <- rbind(dat, temp_dat)
  }
  dat$label <- as.factor(dat$label)

  p <- generic_plot(dat, xaxis) +
    ggplot2::scale_y_continuous(
      #limits=c(max(0, mean(dat$y)-2.5*sd(dat$y)),
      #         min(1, mean(dat$y)+2*sd(dat$y))),
      breaks=seq(0.05, 1, 0.05)) +
    ggplot2::labs(
      title="Classification Error",
      x=xaxis,
      y="")
  return(p)
}

generic_plot <- function(dat, xaxis) {
  x <- NULL
  y <- NULL
  label <- NULL
  p <- ggplot2::ggplot(dat, ggplot2::aes(x=x, y=y, group=label)) +
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
    ggplot2::scale_fill_hue(l=50)
  if (xaxis == "log-Iteration") {
    p <- p +
      ggplot2::scale_x_log10(
        breaks=10^(1:log(max(dat$x), base=10)))
  }
  return(p)
}
