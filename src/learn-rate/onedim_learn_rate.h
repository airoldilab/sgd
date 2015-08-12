#ifndef LEARN_RATE_ONEDIM_LEARN_RATE_H
#define LEARN_RATE_ONEDIM_LEARN_RATE_H

#include "basedef.h"
#include "learn-rate/base_learn_rate.h"
#include "learn-rate/learn_rate_value.h"

class onedim_learn_rate : public base_learn_rate {
  /**
   * One-dimensional (scalar) learning rate, following Xu
   *
   * @param gamma scale factor in both numerator and denominator
   * @param alpha scale factor in denominator
   * @param c     power to exponentiate by
   * @param scale scale factor in numerator
   */
public:
  // Constructors
  onedim_learn_rate(double gamma, double alpha, double c, double scale) :
    gamma_(gamma), alpha_(alpha), c_(c), scale_(scale), v_(0, 1) {}

  // Operators
  virtual const learn_rate_value& operator()(unsigned t, const mat& grad_t) {
    v_ = scale_ * gamma_ * pow(1 + alpha_ * gamma_ * t, -c_);
    return v_;
  }

private:
  double gamma_;
  double alpha_;
  double c_;
  double scale_;
  learn_rate_value v_;
};

#endif
