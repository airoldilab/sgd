#ifndef LEARN_RATE_ONEDIM_EIGEN_LEARN_RATE_H
#define LEARN_RATE_ONEDIM_EIGEN_LEARN_RATE_H

#include "basedef.h"
#include "learn-rate/base_learn_rate.h"
#include "learn-rate/learn_rate_value.h"

class onedim_eigen_learn_rate : public base_learn_rate {
  /**
   * One-dimensional learning rate to parameterize a diagonal matrix
   *
   * @param d  dimension of learning rate
   */
public:
  onedim_eigen_learn_rate(unsigned d) :
    d_(d), v_(0, 1) {}

  virtual const learn_rate_value& operator()(unsigned t, const mat& grad_t) {
    double sum_eigen = 0;
    for (unsigned i = 0; i < d_; ++i) {
      sum_eigen += pow(grad_t.at(i, 0), 2);
    }
    // based on the bound of min_eigen <= d / trace(Fisher_matrix)
    double min_eigen_upper = sum_eigen / d_;
    v_ = 1. / (min_eigen_upper * t);
    return v_;
  }

private:
  unsigned d_;
  learn_rate_value v_;
};

#endif
