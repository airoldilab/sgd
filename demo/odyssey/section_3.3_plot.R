#!/usr/bin/env Rscript
# SECTION 3.3, FIGURE 2
# This is used to generate the plot in M-estimation experiments section.

library(sgd)
library(ggplot2)

generate.data <- function(N, d) {
  l2 <- function(x) sqrt(sum(x**2))
  X <- matrix(rnorm(N*d, mean=0, sd=1/sqrt(N)), nrow=N, ncol=d)
  theta <- runif(d)
  theta <- theta * 6 *sqrt(d) / l2(theta)

  # noise
  ind <- rbinom(N, size=1, prob=.95)
  epsilon <- ind * rnorm(N) + (1-ind) * rep(10 ,N)

  Y <- X %*% theta + epsilon
  return(list(y=Y, X=X, theta=theta))
}

# Dimensions
N <- 100000
d <- 10000

# Generate data.
set.seed(42)
data <- generate.data(N, d)
dat <- data.frame(y=data$y, x=data$X)

sgd.theta1 <- sgd(y ~ .-1, data=dat, model="m",
  sgd.control=list(
  method="sgd",
  lr.control=c(15, NA, NA, 1/2), npass=30, pass=T, size=0.5*N, start=rep(5,d)))
sgd.theta2 <- sgd(y ~ .-1, data=dat, model="m",
  sgd.control=list(
  method="ai-sgd",
  lr.control=c(15, NA, NA, 2/3), npass=30, pass=T, size=0.5*N, start=rep(5,d)))
sgd.theta3 <- sgd(y ~ .-1, data=dat, model="m",
  sgd.control=list(
  method="ai-sgd",
  lr="rmsprop",
  lr.control=c(0.01, NA, NA), npass=30, pass=T, size=0.5*N, start=rep(5,d)))
sgd.theta4 <- sgd(y ~ .-1, data=dat, model="m",
  sgd.control=list(
  method="sgd",
  lr="rmsprop",
  lr.control=c(0.01, NA, NA), npass=30, pass=T, size=0.5*N, start=rep(5,d)))

sgd.list <- list("ai-sgd"=sgd.theta1, "sgd"=sgd.theta2, "sgd+rmsprop"=sgd.theta4,"ai-sgd+rmsprop"=sgd.theta3)

p1 <- plot(sgd.list, data$theta, type="mse-param") +
  geom_hline(yintercept=1.5, color="green") +
  ggplot2::scale_y_continuous(
    breaks=seq(1, 10, 1),
    limits=c(1, 10))
      #legend.position=c(0.55, 0.6),
p2 <- plot(sgd.list, data$theta, type="mse-param", xaxis="runtime") +
  geom_hline(yintercept=1.5, color="green") +
  ggplot2::scale_y_continuous(
    breaks=seq(1, 10, 1),
    limits=c(1, 10))
pdf("temp/huber_1.pdf", width=4, height=4)
print(p1)
dev.off()
pdf("temp/huber_2.pdf", width=4, height=4)
print(p2)
dev.off()
