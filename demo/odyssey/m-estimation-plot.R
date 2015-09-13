#!/usr/bin/env Rscript
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
N <- 10000
d <- 2000

# Generate data.
set.seed(42)
data <- generate.data(N, d)
dat <- data.frame(y=data$y, x=data$X)

sgd.theta1 <- sgd(y ~ .-1, data=dat, model="m",
  sgd.control=list(
  method="sgd",
  lr.control=c(15, NA, NA, 1/2), npass=10, pass=T, size=0.5*N, start=rep(5,d)))
sgd.theta2 <- sgd(y ~ .-1, data=dat, model="m",
  sgd.control=list(
  method="ai-sgd",
  lr.control=c(15, NA, NA, 2/3), npass=10, pass=T, size=0.5*N, start=rep(5,d)))

sgd.list <- list("ai-sgd"=sgd.theta1, "sgd"=sgd.theta2)

p1 <- plot(sgd.list, data$theta, type="mse-param") +
  geom_hline(yintercept=1.5, color="green") +
  ggplot2::scale_y_continuous(
    breaks=seq(1, 10, 1))
p2 <- plot(sgd.list, data$theta, type="mse-param", xaxis="runtime") +
  geom_hline(yintercept=1.5, color="green") +
  ggplot2::scale_y_continuous(
    breaks=seq(1, 10, 1))
pdf("temp/huber_1.pdf", width=4, height=4)
print(p1)
dev.off()
pdf("temp/huber_2.pdf", width=4, height=4)
print(p2)
dev.off()

#library(hqreg)
#time_start <- proc.time()[3]
#hqreg <- hqreg(data$X, as.vector(data$y), method = "huber",
#  gamma=3, alpha=1, lambda=c(1, 0))
#times <- as.numeric(proc.time()[3] - time_start)
