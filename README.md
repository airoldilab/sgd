# sgd

sgd is an R package which provides a fast and flexible set of tools for large
scale inference. It features many different stochastic gradient methods,
built-in models, visualization tools, automated hyperparameter tuning, model
checking, interval estimation, and convergence diagnostics.

## Installation
To install the latest version from CRAN (**under submission**):
```{R}
install.packages("sgd")
```

To install the latest development version from Github:
```{R}
if (packageVersion("devtools") < 1.6) {
  install.packages("devtools")
}
devtools::install_github("airoldilab/sgd")
```

## Features
At the core of the package is the function
```{R}
sgd(formula, data, model, model.control, sgd.control)
```
It implements stochastic gradient descent in order to optimize the underlying
loss function given the data and model; the user can also specify a loss function.

Example of large-scale linear regression:
```{R}
library(sgd)

# Dimensions
N <- 1e5
d <- 1e2

# Generate data.
X <- matrix(rnorm(N*d), ncol=d)
theta <- rep(5, d+1)
eps <- rnorm(N)
y <- cbind(1, X) %*% theta + eps
dat <- data.frame(y=y, x=X)

sgd.theta <- sgd(y ~ ., data=dat, model="lm")
```

The following models are built-in:
* Linear models
* Generalized linear models

The following stochastic gradient methods exist:
* Standard stochastic gradient descent
* Implicit stochastic gradient descent
* Stochastic gradient descent with averaging

For more documentation, see `?sgd`.

## Authors
sgd is written by [Dustin Tran](dtran@g.harvard.edu), [Tian
Lan](tianlan@g.harvard.edu), [Panos Toulis](ptoulis@fas.harvard.edu), and [Ye
Kuang](yekuang@g.harvard.edu), and is in active development. Please feel free
to contribute by submitting any issues or requests—or by solving any current
issues!

We thank all other members of the [Airoldi Lab](http://applied.stat.harvard.edu)
(led by Prof. Edo Airoldi) for their feedback and contributions.
