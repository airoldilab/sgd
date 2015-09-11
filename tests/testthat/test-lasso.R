context("Linear regression with lasso penalty")

test_that("MSE converges for linear regression with lasso", {
  library(glmnet)

  # Dimensions
  N <- 1e5
  d <- 5

  # Generate data.
  set.seed(42)
  X <- matrix(rnorm(N*d), ncol=d)
  theta <- rep(5, d)
  eps <- rnorm(N)
  y <- X %*% theta + eps
  dat <- data.frame(y=y, x=X)

  glmnet.theta <- glmnet(X, y, alpha=1, lambda=0.5, standardize=FALSE,
    type.gaussian="covariance")
  truth <- as.vector(glmnet.theta$beta)

  get.mse <- function(method) {
    sgd.theta <- sgd(y ~ .-1, data=dat, model="lm",
                     model.control=list(lambda1=0.5),
                     sgd.control=list(
                       method=method,
                       pass=T))
    mean((sgd.theta$coefficients - truth)^2)
  }

  expect_true(get.mse("sgd") < 1e-2)
  expect_true(get.mse("ai-sgd") < 1e-2)
})
