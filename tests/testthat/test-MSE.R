context("Mean Squared Error")

test_that("MSE is correct for lm", {
  
  # Dimensions
  N <- 1e4
  d <- 1e1
  
  # Generate data.
  X <- matrix(rnorm(N*d), ncol=d)
  theta <- rep(5, d+1)
  eps <- rnorm(N)
  y <- cbind(1, X) %*% theta + eps
  dat <- data.frame(y=y, x=X)
  
  get.mse <- function(lr_type) {
    sgd.theta <- sgd(y ~ ., data=dat, model="lm", sgd.control=list(lr=lr_type))
    mean((sgd.theta$coefficients - theta)^2)
  } 
  
  expect_true(get.mse('one-dim') < 1e-2)
  expect_true(get.mse('one-dim-eigen') < 1e-2)
  expect_true(get.mse('adagrad') < 1e-2)
  expect_true(get.mse('d-dim') < 1e-2)
})
