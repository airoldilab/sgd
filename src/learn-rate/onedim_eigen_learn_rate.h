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
   *
   * @param d  dimension of learning rate
   * @param gr gradient function
   */
public:
  // Constructors
  onedim_eigen_learn_rate(unsigned d, const grad_func_type& gr) :
    d_(d), grad_func_(gr), v_(0, 1) {}

  // Operators
  virtual const learn_rate_value& operator()(const mat& theta_old, const
    data_point& data_pt, unsigned t) {
    mat Gi = grad_func_(theta_old, data_pt);
    double sum_eigen = 0;
    for (unsigned i = 0; i < d_; ++i) {
      sum_eigen += pow(Gi.at(i, 0), 2);
    }
    // based on the bound of min_eigen <= d / trace(Fisher_matrix)
    double min_eigen_upper = sum_eigen / d_;
    v_ = 1. / (min_eigen_upper * t);
    return v_;
  }

private:
  unsigned d_;
  grad_func_type grad_func_;
  learn_rate_value v_;
};

#endif
