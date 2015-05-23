#ifndef BASEDEF_H
#define BASEDEF_H

// In unittest, switch this to 0
#define __R__ 1
#define DEBUG 0

#if __R__
	#include "RcppArmadillo.h"
#else
	#include <armadillo>
#endif

#include <boost/function.hpp>
#include <boost/tuple/tuple.hpp>
#include <math.h>
#include <string>
#include <cstddef>

using namespace arma;

typedef boost::function<double (double)> uni_func_type;
typedef boost::function<mat (const mat&)> mmult_func_type;
typedef boost::function<double (const mat&, const mat&, const mat&)> deviance_type;

#endif
