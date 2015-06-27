#ifndef LEARN_RATE_ONEDIM_EIGEN_LEARN_RATE_H
#define LEARN_RATE_ONEDIM_EIGEN_LEARN_RATE_H

#include "basedef.h"
#include "data/data_point.h"
#include "learn-rate/base_learn_rate.h"
#include "learn-rate/learn_rate_value.h"

typedef boost::function<mat(const mat&, const data_point&)> grad_func_type;

class onedim_eigen_learn_rate : public base_learn_rate {
  /**
   * One-dimensional learning rate to parameterize a diagonal matrix
   */
public:
  // Constructors
  onedim_eigen_learn_rate(const grad_func_type& gr) : grad_func(gr), v(0, 1) {}

  // Operators
  virtual const learn_rate_value& operator()(const mat& theta_old, const
    data_point& data_pt, unsigned t, unsigned d) {
    mat Gi = grad_func(theta_old, data_pt);
    double sum_eigen = 0;
    for (unsigned i = 0; i < d; ++i) {
      sum_eigen += pow(Gi.at(i, 0), 2);
    }
    // based on the bound of min_eigen <= d / trace(Fisher_matrix)
    double min_eigen_upper = sum_eigen / d;
    v.lr_scalar = 1. / (min_eigen_upper * t);
    return v;
  }

private:
  grad_func_type grad_func;
  learn_rate_value v;
};

#endif
