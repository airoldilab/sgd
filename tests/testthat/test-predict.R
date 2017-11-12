context("Predict method")

test_that("Predict method", {

  skip_on_cran()

  # Dimensions
  N <- 1e4
  d <- 5

  # Generate data.
  set.seed(42)
  X <- matrix(rnorm(N*d), ncol=d)
  theta <- rep(5, d+1)
  eps <- rnorm(N)
  y <- cbind(1, X) %*% theta + eps
  dat <- data.frame(y=y, x=X)

  sgd.theta <- sgd(y ~ ., data=dat, model="lm")
  predict(sgd.theta, cbind(1, X))
  predict(sgd.theta, cbind(1, X), type="response")
  predict(sgd.theta, cbind(1, X), type="term")

  expect_true(TRUE)
})
