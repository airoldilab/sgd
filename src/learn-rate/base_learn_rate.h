#ifndef LEARN_RATE_BASE_LEARN_RATE_H
#define LEARN_RATE_BASE_LEARN_RATE_H

#include "basedef.h"
#include "data/data_point.h"
#include "learn-rate/learn_rate_value.h"

using namespace arma;

class base_learn_rate {
  /* Base class from which all learning rate classes inherit from */
public:
#if DEBUG
  virtual ~base_learn_rate() {
    Rcpp::Rcout << "Learning rate object released" << std::endl;
  }
#else
  virtual ~base_learn_rate() {}
#endif
  virtual const learn_rate_value& learning_rate(const mat& theta_old, const
    data_point& data_pt, double offset, unsigned t, unsigned d) = 0;
};

#endif
