# sgd

sgd is an R package for large
scale estimation. It features many stochastic gradient methods, built-in models,
visualization tools, automated hyperparameter tuning, model checking, interval
estimation, and convergence diagnostics.

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
* Method of moments
* Generalized method of moments
* Cox proportional hazards model
* M-estimation

The following stochastic gradient methods exist:
* (Standard) stochastic gradient descent
* Implicit stochastic gradient descent
* Averaged stochastic gradient descent
* Averaged implicit stochastic gradient descent
* Classical momentum
* Nesterov's accelerated gradient

Check out the vignette in the `vignettes/` directory, or examples in `demo/`.
In R, the equivalent commands are `vignette(package="sgd")` and
`demo(package="sgd")`.

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

## Authors
sgd is written by [Dustin Tran](http://dustintran.com) and
[Panos Toulis](http://www.people.fas.harvard.edu/~ptoulis), and is under active
development. Please feel free to contribute by submitting any issues or
requests—or by solving any current issues!

We thank all other members of the [Airoldi Lab](http://applied.stat.harvard.edu)
(led by Prof. Edo Airoldi) for their feedback and contributions.

## Citation

```
@article{tran2015stochastic,
  author = {Tran, Dustin and Toulis, Panos and Airoldi, Edoardo M},
  title = {Stochastic gradient descent methods for estimation with large data sets},
  journal = {arXiv preprint arXiv:1509.06459},
  year = {2015}
}
```
