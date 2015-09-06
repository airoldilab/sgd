# sgd

sgd is an R package which provides a fast and flexible set of tools for large
scale inference. It features many stochastic gradient methods, built-in models,
visualization tools, automated hyperparameter tuning, model checking, interval
estimation, and convergence diagnostics.

## Installation
To install the latest version from CRAN:
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
It estimates parameters for a given data set and model using stochastic gradient
descent. The optional arguments `model.control` and `sgd.control` specify
attributes about the model and stochastic gradient method. Taking advantage of
the bigmemory package, sgd also operates on data sets which are too large to fit
in RAM as well as streaming data.

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

Any loss function may be specified, although for convenience the following are
built-in:
* Linear models
* Generalized linear models
* Generalized method of moments
* Cox proportional hazards model
* M-estimation

The following stochastic gradient methods exist:
* Standard stochastic gradient descent
* Implicit stochastic gradient descent
* Stochastic gradient descent with averaging
* Implicit stochastic gradient descent with averaging
* Classical momentum
* Nesterov's accelerated gradient

For more examples, see the `demo/` directory, and for more documentation, run
`?sgd` or `library(help=sgd)` in R.

## Authors
sgd is written by [Dustin Tran](http://dustintran.com), [Tian
Lan](mailto:tianlan@g.harvard.edu), [Panos
Toulis](http://www.people.fas.harvard.edu/~ptoulis), and [Ye
Kuang](mailto:yekuang@g.harvard.edu), and is under active development. Please
feel free to contribute by submitting any issues or requests—or by solving any
current issues!

We thank all other members of the [Airoldi Lab](http://applied.stat.harvard.edu)
(led by Prof. Edo Airoldi) for their feedback and contributions.
