context("Mean Squared Error")

test_that("MSE converges for linear models", {

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

  get.mse <- function(method, lr) {
    sgd.theta <- sgd(y ~ ., data=dat, model="lm",
                     sgd.control=list(method=method, lr=lr,
                     npasses=10))
    mean((sgd.theta$coefficients - theta)^2)
  }

  # TODO
  # one-dim-eigen is bugged
  # d-dim is bugged for explicit methods
  expect_true(get.mse("sgd", "one-dim") < 1e-2)
  #expect_true(get.mse("sgd", "one-dim-eigen") < 1e-2)
  #expect_true(get.mse("sgd", "d-dim") < 1e-2)
  expect_true(get.mse("sgd", "adagrad") < 1e-2)
  expect_true(get.mse("sgd", "rmsprop") < 1e-2)
  expect_true(get.mse("implicit", "one-dim") < 1e-2)
  #expect_true(get.mse("implicit", "one-dim-eigen") < 1e-2)
  expect_true(get.mse("implicit", "d-dim") < 1e-2)
  expect_true(get.mse("implicit", "adagrad") < 1e-2)
  expect_true(get.mse("implicit", "rmsprop") < 1e-2)
  expect_true(get.mse("asgd", "one-dim") < 1e-2)
  #expect_true(get.mse("asgd", "one-dim-eigen") < 1e-2)
  #expect_true(get.mse("asgd", "d-dim") < 1e-2)
  expect_true(get.mse("asgd", "adagrad") < 1e-2)
  expect_true(get.mse("asgd", "rmsprop") < 1e-2)
  expect_true(get.mse("ai-sgd", "one-dim") < 1e-2)
  #expect_true(get.mse("ai-sgd", "one-dim-eigen") < 1e-2)
  expect_true(get.mse("ai-sgd", "d-dim") < 1e-2)
  expect_true(get.mse("ai-sgd", "adagrad") < 1e-2)
  expect_true(get.mse("ai-sgd", "rmsprop") < 1e-2)
})
