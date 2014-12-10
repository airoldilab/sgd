library(Rcpp)
library(microbenchmark)

sourceCpp("matMultCpp.cpp")

m1 <- array(2, dim=c(100, 200))
m2 <- array(1, dim=c(200,100))

matMultR <- function(m1, m2) {
  out <- m1 %*% m2
  out
}

# extremely bad example...... 
# simple case to illustrate how to use NumericMatrix in C++
microbenchmark(
  matMultR(m1, m2),
  matMultCpp(m1, m2)
)