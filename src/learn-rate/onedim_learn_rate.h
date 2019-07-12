#ifndef LEARN_RATE_ONEDIM_LEARN_RATE_H
#define LEARN_RATE_ONEDIM_LEARN_RATE_H

#include "../basedef.h"
#include "base_learn_rate.h"
#include "learn_rate_value.h"

class onedim_learn_rate : public base_learn_rate {
  /**
   * One-dimensional (scalar) learning rate, following Xu
   *
   * @param scale scale factor in numerator
   * @param gamma scale factor in both numerator and denominator
   * @param alpha scale factor in denominator
   * @param c     power to exponentiate by
   */
public:
  onedim_learn_rate(double scale, double gamma, double alpha, double c) :
    scale_(scale), gamma_(gamma), alpha_(alpha), c_(c), v_(0, 1) {}

  virtual const learn_rate_value& operator()(unsigned t, const mat& grad_t) {
    v_ = scale_ * gamma_ * pow(1 + alpha_ * gamma_ * t, -c_);
    return v_;
  }

private:
  double scale_;
  double gamma_;
  double alpha_;
  double c_;
  learn_rate_value v_;
};

#endif
