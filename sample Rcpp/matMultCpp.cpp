#include <Rcpp.h>
#include <math.h>
#include <stdexcept>

using namespace Rcpp;

typedef long size_type;
typedef NumericVector vec_type;
typedef NumericMatrix mat_type;

// Below is a simple example of exporting a C++ function to R. You can
// source this function into an R session using the Rcpp::sourceCpp 
// function (or via the Source button on the editor toolbar)

// For more on using Rcpp click the Help button on the editor toolbar


// [[Rcpp::export]]
mat_type matMultCpp(mat_type& m1, mat_type& m2) {
	mat_type out(m1.nrow(), m2.ncol());

	for (size_type i = 0; i < m1.nrow(); ++i) {
		for(size_type j = 0; j < m2.ncol(); ++j) {
			out(i, j) = 0;
			for (size_type k = 0; k < m1.ncol(); ++k) {
				out(i, j) += m1(i, k) * m2(k, j);
			}
		}
	}
	return out;
}
