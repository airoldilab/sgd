sgd-r-package
=============

The R package of stochastic gradient descent(SGD) with explicit and implicit updates for generalized linear models.

### File Description

### How to install
To install the package locally:
```{bash}
install.packages(path_to_file, repos = NULL, type="source")
```

Note:
* As the package is installed for local source, dependencies will not be automatically installed.

Dependencies: MASS, Rcpp, RcppArmadillo, and BH

#### testcpp  
This is the C++ implementation of the accompanying code of the methods and algorithms 
presented
```
Panos Toulis, Jason Rennie, Edoardo Airoldi, 
"Statistical analysis of stochastic gradient methods for generalized linear models", 
ICML, Beijing, China, 2014.
```
