#ifndef BASEDEF_H
#define BASEDEF_H
#define BOOST_DISABLE_ASSERTS true

// In unittest, switch this to 0
#define __R__ 1
#define DEBUG 0

#if __R__
  #include "RcppArmadillo.h"
#else
  #include <armadillo>
#endif

#include <boost/function.hpp>
#include <boost/timer.hpp>
#include <boost/tuple/tuple.hpp>
#include <math.h>
#include <string>
#include <cstddef>
#include <bigmemory/MatrixAccessor.hpp>

using namespace arma;

#endif
