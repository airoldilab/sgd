#' Plot objects of class \code{sgd}.
#'
#' @param x object of class \code{sgd}.
#' @param \dots additional arguments used for each type of plot. See
#'   \sQuote{Details}.
#' @param type character specifying the type of plot: \code{"mse"},
#'   \code{"runtime"}. See \sQuote{Details}.
#'
#' @details
#' Types:
#' \describe{
#'   \item{\code{mse}}{Mean squared error in predictions, which takes the
#'     following arguments:
#'     \describe{
#'       \item{\code{x_test}}{test set}
#'       \item{\code{y_test}}{test responses to compare predictions to}
#'     }}
#'   \item{\code{runtime}}{Mean squared error in predictions over runtime, which
#'     takes the following arguments:
#'     \describe{
#'       \item{\code{x_test}}{test set}
#'       \item{\code{y_test}}{test responses to compare predictions to}
#'     }}
#'   \item{\code{mse-param}}{Mean squared error in parameters, which takes the
#'     following arguments:
#'     \describe{
#'       \item{\code{true_param}}{true vector of parameters to compare to}
#'     }}
#'   \item{\code{mse-param-runtime}}{Mean squared error in parameters over
#'     runtime, which takes the following arguments:
#'     \describe{
#'       \item{\code{true_param}}{true vector of parameters to compare to}
#'     }}
#' }
#'
#' @export
plot.sgd <- function(x, ..., type="mse") {
  if (type == "mse") {
    plot <- plot_mse
  } else if (type %in% c("runtime", "mse-runtime")) {
    plot <- function(x, ...) plot_mse(x, ..., xaxis="Runtime (s)")
  } else if (type == "mse-param") {
    plot <- plot_mse_param
  } else if (type == "mse-param-runtime") {
    plot <- function(x, ...) plot_mse_param(x, ..., xaxis="Runtime (s)")
  } else {
    stop("'type' not recognized")
  }
  return(plot(x, ...))
}

#' @export
#' @rdname plot.sgd
plot.list <- function(x, ..., type="mse") {
  if (type == "mse") {
    plot <- plot_mse
  } else if (type %in% c("runtime", "mse-runtime")) {
    plot <- function(x, ...) plot_mse(x, ..., xaxis="Runtime (s)")
  } else if (type == "mse-param") {
    plot <- plot_mse_param
  } else if (type == "mse-param-runtime") {
    plot <- function(x, ...) plot_mse_param(x, ..., xaxis="Runtime (s)")
  } else {
    stop("'type' not recognized")
  }
  return(plot(x, ...))
}

################################################################################
# Helper functions
################################################################################

get_mse_glm <- function(x, x_test, y_test) {
  nests <- ncol(x$estimates)
  eta <- x_test %*% x$estimates # assuming intercepts in X
  mu <- x$model.out$family$linkinv(eta)
  mse <- rep(NA, nests)
  for (j in 1:nests) {
    mse[j] <- mean((mu[, j] - y_test)^2)
  }
  return(mse)
}

get_mse_m <- function(x, x_test, y_test) {
  nests <- ncol(x$estimates)
  eta <- x_test %*% x$estimates # assuming intercepts in X
  mu <- eta
  mse <- rep(NA, nests)
  for (j in 1:nests) {
    mse[j] <- mean((mu[, j] - y_test)^2)
  }
  return(mse)
}

plot_mse <- function(x, x_test, y_test, label=1, xaxis="log-Iteration") {
  if (x$model %in% c("lm", "glm")) {
    get_mse <- get_mse_glm
  } else if (x$model == "m") {
    get_mse <- get_mse_m
  # TODO
  } else {
    stop("'model' not recognized")
  }

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
    ggplot2::scale_y_log10()
  return(p)
}

get_mse_param <- function(x, true_param) {
  nests <- ncol(x$estimates)
  mse <- rep(NA, nests)
  for (j in 1:nests) {
    mse[j] <- mean((x$estimates[, j] - true_param)^2)
  }
  return(mse)
}

plot_mse_param <- function(x, true_param, label=1, xaxis="log-Iteration") {
  get_mse <- get_mse_param

  if (class(x) != "list") {
    x <- list(x)
    names(x) <- label
  }
  dat <- data.frame()
  for (i in 1:length(x)) {
    mse <- get_mse(x[[i]], true_param)
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
      breaks=5 * 1:min((max(dat$y)/5), 100))
  return(p)
}

generic_plot <- function(dat, xaxis) {
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
    ggplot2::scale_fill_hue(l=50) +
    ggplot2::labs(
      title="Mean Squared Error",
      x=xaxis,
      y=""
    )
  if (xaxis == "log-Iteration") {
    p <- p +
      ggplot2::scale_x_log10(
        breaks=10^(1:log(max(dat$x), base=10)))
  }
  return(p)
}
