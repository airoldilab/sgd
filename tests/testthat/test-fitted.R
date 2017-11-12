context("Fitted generic method")

test_that("Fitted generic method", {

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
  fitted(sgd.theta)

  # Check that it executes without error.
  expect_true(TRUE)
})
