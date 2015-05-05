# sgd

sgd is an R package for performing inference on large scale data sets. It
optimizes the underlying loss function using stochastic gradient descent (SGD).
The package contains several variants of the method, e.g., explicit and implicit
updates, L1/L2 regularization, averaging, and furthermore it comes integrated
with several built-in models:

* Generalized linear models (GLMs)
* Linear support vector machines (SVMs)
* Cox proportional hazards model
* Generalized additive models (GAMs)
* ...

## Installation
Dependencies: `MASS`, `Rcpp`, `RcppArmadillo`, `BH`

The latest development version from Github can be installed using the `devtools`
library.
```{R}
if (packageVersion("devtools") < 1.6) {
  install.packages("devtools")
}
devtools::install_github("airoldilab/sgd")
```
**Note that as this is currently a private repository, one must generate an
authorization token to install from Github**. [Generate a personal access token
(PAT)](https://github.com/settings/applications) and supply it as an argument to
`auth_token`:
```{R}
devtools::install_github("airoldilab/sgd", auth_token="auth_token_here")
```

To install locally, one must install the dependencies manually and zip this repo
into a `.tar.gz` file. Then run
```{R}
install.packages(path_to_file, repos=NULL, type="source")
```

## Maintainers
* Tian Lan \<tianlan@g.harvard.edu\>
* Dustin Tran \<dtran@g.harvard.edu\>
* Panos Toulis \<ptoulis@fas.harvard.edu\>

## References
* Boris T. Polyak and Anatoli B. Juditsky. Acceleration of stochastic
  approximation by averaging. _SIAM Journal on Control and Optimization_,
  30(4):838–855, 1992.
* Herbert Robbins and Sutton Monro. A stochastic approximation method. _The
  Annals of Mathematical Statistics_, pp. 400–407, 1951.
* Panos Toulis, Jason Rennie, and Edoardo M. Airoldi, "Statistical analysis of
  stochastic gradient methods for generalized linear models", In _Proceedings of
  the 31st International Conference on Machine Learning (ICML)_, 2014.
